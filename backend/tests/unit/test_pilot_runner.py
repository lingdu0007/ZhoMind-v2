import httpx
import pytest

from tests.unit.test_pilot_measurement import binding


def test_entry_runner_rejects_missing_server_snapshot_fields_before_load():
    from app.operations.pilot_runner import run_entry_requests

    def respond(request):
        if request.url.path.endswith("/auth/me"):
            return httpx.Response(200, json={"data": {"role": "user", "username": request.headers["authorization"]}})
        if request.url.path.endswith("/measurement-snapshot"):
            return httpx.Response(200, json={"data": {"background_workers": 1}})
        raise AssertionError("incomplete snapshot reached workload execution")

    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://local.test") as client:
        with pytest.raises(ValueError, match="snapshot"):
            run_entry_requests(client, binding=binding(), member_tokens=tuple(f"member-{i}" for i in range(8)),
                               administrator_token="admin", question="not-sent")
