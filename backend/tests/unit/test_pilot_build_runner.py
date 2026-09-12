from datetime import UTC, datetime, timedelta

import httpx
import pytest

from tests.unit.test_pilot_measurement import binding


@pytest.mark.parametrize("count", [10, 50])
@pytest.mark.parametrize("naive_utc", [False, True])
def test_bundle_measurement_dispatches_exact_items_and_retains_only_identity_and_timings(count, naive_utc):
    from app.common.canonical_json import canonical_json_sha256
    from app.operations.pilot_build_runner import measure_bundle

    now = datetime.now(UTC)
    revisions = [(f"entry:{i}", f"editorial_revision:{i}.r1") for i in range(count)]
    bound = binding(window_start=now, window_end=now + timedelta(hours=1),
                    editorial_inputs_sha256=canonical_json_sha256(sorted(revisions)))
    dispatched = set()
    manifest = {"items": [{"operation": "replace" if count == 50 else "create", "artifact": {
        "entry_identity": entry, "editorial_revision_identity": revision,
    }} for entry, revision in revisions], "bundle_sha256": "a" * 64}

    def respond(request):
        path = request.url.path
        if path.endswith("/auth/me"):
            return httpx.Response(200, json={"data": {"username": "measurement-admin", "role": "admin"}})
        if path.endswith("/import"):
            return httpx.Response(200, json={"data": {
                "bundle_sha256": "a" * 64,
                "items": [{"job_id": str(i), "bundle_item_sha256": f"{i:064x}"} for i in range(count)],
            }})
        if path.endswith("/dispatch"):
            dispatched.add(path.split("/")[-2])
            return httpx.Response(200, json={"data": {}})
        if "/jobs/" in path:
            job = path.split("/")[-1]
            instant = datetime.now(UTC)
            stamp = (instant.replace(tzinfo=None) if naive_utc else instant).isoformat()
            return httpx.Response(200, json={"data": {
                "job_id": job, "status": "candidate_ready" if job in dispatched else "queued", "attempt": 1,
                "dispatched_at": None, "started_at": stamp,
                "completed_at": stamp,
                "candidate_id": f"candidate:{job}", "frozen_input_sha256": f"{int(job):064x}",
                "embedding_configuration": {"schema": "candidate_embedding_configuration/v1"},
            }})
        if path.endswith("/inspection"):
            return httpx.Response(200, json={"data": {"candidate": {
                "chunks": [{"content": "PRIVATE_BUILD_BODY", "content_sha256": "c" * 64}],
            }}})
        raise AssertionError("unexpected product operation")

    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://local.test") as client:
        result = measure_bundle(client, binding=bound, administrator_token="private-local-token",
                                manifest=manifest, kind="ten_item_bundle" if count == 10 else "fifty_entry_rebuild")
    assert len(dispatched) == count
    assert result["complete_items"] == count
    assert result["status"] == "passing"
    assert "tightening_decisions" in result
    assert "PRIVATE_BUILD_BODY" not in str(result)
    assert "private-local-token" not in str(result)


def test_no_op_is_not_counted_as_a_reimport_rebuild():
    from app.operations.pilot_build_runner import measure_bundle

    with pytest.raises(ValueError, match="fresh build"):
        measure_bundle(None, binding=binding(), administrator_token="not-used",
                       manifest={"items": [{"operation": "no_op"} for _ in range(50)]}, kind="fifty_entry_rebuild")


def test_build_cannot_execute_under_a_past_observation_window():
    from app.operations.pilot_build_runner import measure_bundle

    with pytest.raises(ValueError, match="observation window"):
        measure_bundle(None, binding=binding(), administrator_token="not-used",
                       manifest={"items": [{"operation": "create"} for _ in range(10)]}, kind="ten_item_bundle")


@pytest.mark.parametrize("operation", ["create", "replace"])
def test_fifty_foreign_entries_cannot_replace_the_bound_rebuild_corpus(operation):
    from app.operations.pilot_build_runner import measure_bundle

    now = datetime.now(UTC)
    bound = binding(window_start=now, window_end=now + timedelta(hours=1))
    manifest = {"items": [{"operation": operation, "artifact": {
        "entry_identity": f"entry:foreign-{index}", "editorial_revision_identity": f"editorial_revision:foreign-{index}.r1",
    }} for index in range(50)]}
    with pytest.raises(ValueError, match="bound corpus"):
        measure_bundle(None, binding=bound, administrator_token="not-used", manifest=manifest, kind="fifty_entry_rebuild")
