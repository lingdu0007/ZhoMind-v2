import asyncio

import pytest
from sqlalchemy import delete, select, update

from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket27_maintenance import _maintainer


@pytest.mark.parametrize(
    "corruption",
    [
        "schema", "record_class", "record_state", "unknown_record_field", "assigner",
        "unknown_event_field", "event_kind", "event_from_state", "duplicate_acceptance", "missing_acceptance",
    ],
)
def test_malformed_assignment_cannot_authorize_maintenance_or_report_successful_acceptance(client, corruption):
    from app.model.canonical import CanonicalEventModel, CanonicalRecordModel

    admin, maintainer, _worker = _maintainer(client)
    context = client.get("/api/v1/maintenance/context", headers=maintainer).json()["data"]
    identity = context["assignment"]["id"]
    assert context["is_maintainer"] is True

    async def corrupt():
        async with client.app.state.test_auth_session_factory() as session:
            connection = await session.connection()
            record = await session.get(CanonicalRecordModel, identity)
            assert record is not None
            if corruption in {"schema", "unknown_record_field", "assigner"}:
                patch = {
                    "schema": {"schema": "unrelated/v1"},
                    "unknown_record_field": {"note": "private assignment corruption"},
                    "assigner": {"assigned_by": context["member_identity"].removeprefix("member:")},
                }[corruption]
                await connection.execute(
                    update(CanonicalRecordModel).where(CanonicalRecordModel.stable_id == identity)
                    .values(payload={**record.payload, **patch}),
                )
            elif corruption in {"record_class", "record_state"}:
                column = "record_class" if corruption == "record_class" else "state"
                await connection.execute(
                    update(CanonicalRecordModel).where(CanonicalRecordModel.stable_id == identity)
                    .values(**{column: "unverified"}),
                )
            else:
                event = await session.scalar(
                    select(CanonicalEventModel).where(
                        CanonicalEventModel.aggregate_id == identity,
                        CanonicalEventModel.event_type == "responsibility_accepted",
                    ),
                )
                assert event is not None
                if corruption == "missing_acceptance":
                    await connection.execute(delete(CanonicalEventModel).where(CanonicalEventModel.id == event.id))
                    return await session.commit()
                if corruption == "duplicate_acceptance":
                    session.add(CanonicalEventModel(
                        aggregate_id=identity, aggregate_kind=event.aggregate_kind,
                        event_type=event.event_type, from_state=event.from_state, to_state=event.to_state,
                        recorded_by=event.recorded_by, payload=event.payload,
                    ))
                else:
                    patch = {
                        "unknown_event_field": {"payload": {**event.payload, "note": "private assignment corruption"}},
                        "event_kind": {"aggregate_kind": "entry"},
                        "event_from_state": {"from_state": "revoked"},
                    }[corruption]
                    await connection.execute(
                        update(CanonicalEventModel).where(CanonicalEventModel.id == event.id).values(**patch),
                    )
            await session.commit()

    asyncio.run(corrupt())
    inbox = client.get("/api/v1/maintenance/inbox", headers=maintainer)
    assert inbox.status_code == 403, inbox.text
    assert "private assignment corruption" not in inbox.text
    current = client.get("/api/v1/maintenance/context", headers=maintainer).json()["data"]
    assert current["is_maintainer"] is False
    if corruption != "missing_acceptance":
        assert current["assignment"] is None
        accepted = client.post(f"/api/v1/maintenance/assignments/{identity}/accept", headers=maintainer)
        assert accepted.status_code == 403, accepted.text
        assigned = client.post("/api/v1/maintenance/assignments", headers=admin, json={"username": "maintenance-editor"})
        assert assigned.status_code == 403, assigned.text


def test_valid_assignment_and_acceptance_retries_preserve_one_acceptance_event(client):
    from app.model.canonical import CanonicalEventModel

    admin, maintainer, worker = _maintainer(client)
    context = client.get("/api/v1/maintenance/context", headers=maintainer).json()["data"]
    identity = context["assignment"]["id"]
    for _ in range(2):
        assigned = client.post("/api/v1/maintenance/assignments", headers=admin, json={"username": "maintenance-editor"})
        assert assigned.status_code == 200, assigned.text
        assert assigned.json()["data"]["id"] == identity
        accepted = client.post(f"/api/v1/maintenance/assignments/{identity}/accept", headers=maintainer)
        assert accepted.status_code == 200, accepted.text
    assert client.post(f"/api/v1/maintenance/assignments/{identity}/accept", headers=worker).status_code == 403
    assert client.get("/api/v1/maintenance/inbox", headers=maintainer).status_code == 200

    async def events():
        async with client.app.state.test_auth_session_factory() as session:
            return list((await session.scalars(
                select(CanonicalEventModel).where(CanonicalEventModel.aggregate_id == identity),
            )).all())

    assert len(asyncio.run(events())) == 1


@pytest.mark.parametrize("corruption", ["missing_owner", "unhashable_owner", "record_class", "identity_kind", "identity_value"])
def test_malformed_item_authority_is_rejected_before_owner_projection(client, corruption):
    from app.model.canonical import CanonicalRecordModel
    from tests.integration.test_ticket27_maintenance import _reproduction_case

    maintainer, _worker, _reporter, base, _reproduction = _reproduction_case(client)
    identity = base.rsplit("/", 1)[1]

    async def corrupt():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, identity)
            assert record is not None
            if corruption in {"missing_owner", "unhashable_owner"}:
                payload = dict(record.payload)
                if corruption == "missing_owner":
                    payload.pop("work_owner")
                else:
                    payload["work_owner"] = {"private": "must not project"}
                patch = {"payload": payload}
            else:
                patch = {corruption: "unverified"}
            connection = await session.connection()
            await connection.execute(
                update(CanonicalRecordModel).where(CanonicalRecordModel.stable_id == identity).values(**patch),
            )
            await session.commit()

    asyncio.run(corrupt())
    result = client.get(base, headers=maintainer)
    assert result.status_code == 409, result.text
    assert result.json()["code"] == "MAINTENANCE_EVENT_INVALID"
    assert "must not project" not in result.text
    if corruption in {"missing_owner", "unhashable_owner", "record_class"}:
        listing = client.get("/api/v1/maintenance/items", headers=maintainer)
        assert listing.status_code == 409, listing.text
        assert listing.json()["code"] == "MAINTENANCE_EVENT_INVALID"
