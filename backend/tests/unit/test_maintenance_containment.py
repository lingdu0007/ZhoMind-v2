from uuid import uuid4

import pytest

from app.common.exceptions import AppError
from app.maintenance.containment import verification_scope
from app.model.user import User


@pytest.mark.parametrize("severity", ["p0", "p1"])
@pytest.mark.parametrize("corruption", ["event", "scope", "missing_scope"])
def test_malformed_verification_authority_is_a_structured_denial(severity, corruption):
    item = {
        "id": f"maintenance_item:{uuid4().hex}",
        "severity": severity,
        "revision": 2,
        "containment_event_id": uuid4().hex,
        "blocking_scope": {"scope": "entry_version", "identity": "entry:contained-entry"},
    }
    if corruption == "event":
        item["containment_event_id"] = "invalid-event"
    elif corruption == "scope":
        item["blocking_scope"] = {"scope": "entry_version", "identity": "collection:unrelated"}
    else:
        item["blocking_scope"] = None
    actor = User(id=uuid4(), username="verification-worker", role="user", is_active=True)
    with pytest.raises(AppError) as failure:
        with verification_scope(item, actor):
            pytest.fail("malformed authority cannot enter verification")
    assert failure.value.status_code == 409
    assert failure.value.code == "MAINTENANCE_EVENT_INVALID"
