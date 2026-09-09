from copy import deepcopy
from pathlib import Path

import pytest
import yaml


def test_supported_compose_exposes_only_https_edge():
    from app.operations.exposure import inspect_exposure

    path = Path(__file__).resolve().parents[3] / "deploy/production/compose.yml"
    config = yaml.safe_load(path.read_text())
    report = inspect_exposure(config)
    assert report["configuration_passed"] is True
    assert report["live_exposure_verified"] is False
    assert report["public_services"] == ["caddy"]
    assert "environment" not in str(report)
    for service in ("postgres", "redis", "etcd", "minio", "milvus", "backend"):
        unsafe = deepcopy(config)
        unsafe["services"][service]["ports"] = ["5432:5432"]
        assert inspect_exposure(unsafe)["configuration_passed"] is False


def test_supported_edge_can_omit_http3_without_exposing_other_ports():
    from app.operations.exposure import inspect_exposure

    path = Path(__file__).resolve().parents[3] / "deploy/production/compose.yml"
    config = yaml.safe_load(path.read_text())
    config["services"]["caddy"]["ports"] = ["80:80", "443:443"]
    assert inspect_exposure(config)["configuration_passed"] is True
    for extra in ("2019:2019", "8000:8000", "443:443/udp", "8443:443"):
        unsafe = deepcopy(config)
        unsafe["services"]["backend"]["ports"] = [extra]
        assert inspect_exposure(unsafe)["configuration_passed"] is False
    config["services"]["caddy"]["ports"].append("2019:2019")
    assert inspect_exposure(config)["configuration_passed"] is False


@pytest.mark.parametrize("status,location,expected", [
    (308, "https://kb.example.com/chat", True),
    (200, None, False),
    (302, "https://foreign.example.com/chat", False),
    (308, "http://kb.example.com/chat", False),
    (403, None, True),
])
def test_plaintext_probe_accepts_only_rejection_or_same_origin_https(status, location, expected):
    from app.operations.exposure import plaintext_is_closed

    assert plaintext_is_closed(status, location, "kb.example.com") is expected


@pytest.mark.parametrize("error,expected", [(ConnectionRefusedError, True), (TimeoutError, False)])
def test_exposure_distinguishes_closed_ports_from_incomplete_probes(monkeypatch, error, expected):
    import socket
    from functools import partial

    import httpx

    from app.operations.exposure import probe_exposure

    def respond(request):
        if request.url.scheme == "http":
            return httpx.Response(308, headers={"location": "https://kb.example.com/"})
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"data": {"status": "up"}})
        return httpx.Response(200, text='<div id="app"></div>')

    def connect(*args, **kwargs):
        raise error()

    monkeypatch.setattr(httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443)),
    ])
    monkeypatch.setattr(socket, "create_connection", connect)
    result = probe_exposure("kb.example.com")
    assert result["tcp_probe_complete"] is expected
    assert result["live_exposure_verified"] is expected


def test_exposure_can_bound_simultaneous_external_connections(monkeypatch):
    import socket
    import threading
    import time
    from functools import partial

    import httpx

    from app.operations.exposure import probe_exposure

    def respond(request):
        if request.url.scheme == "http":
            return httpx.Response(308, headers={"location": "https://kb.example.com/"})
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"data": {"status": "up"}})
        return httpx.Response(200, text='<div id="app"></div>')

    active = 0
    peak = 0
    attempts = []
    lock = threading.Lock()

    def connect(address, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            attempts.append(address)
        try:
            time.sleep(0.002)
            raise ConnectionRefusedError()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443)),
    ])
    monkeypatch.setattr(socket, "create_connection", connect)
    result = probe_exposure("kb.example.com", tcp_workers=1)
    assert result["live_exposure_verified"] is True
    assert peak == 1
    assert {port for _, port in attempts} == {
        2019, 2379, 2380, 3000, 5432, 6379, 8000, 9000, 9001, 9090, 9091, 19530,
    }


@pytest.mark.parametrize("workers", [0, 65, True, "2"])
def test_exposure_rejects_invalid_concurrency_before_network_access(monkeypatch, workers):
    import httpx

    from app.operations.exposure import probe_exposure

    def unexpected_network(*args, **kwargs):
        pytest.fail("Invalid probe parameters must not initiate network access")

    monkeypatch.setattr(httpx, "Client", unexpected_network)
    with pytest.raises(ValueError, match="tcp_workers"):
        probe_exposure("kb.example.com", tcp_workers=workers)
