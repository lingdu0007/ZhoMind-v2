import argparse
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import yaml

from app.common.canonical_json import canonical_json_sha256

_SERVICES = {"postgres", "redis", "etcd", "minio", "milvus", "backend", "caddy"}
_PRIVATE_PORTS = (2019, 2379, 2380, 3000, 5432, 6379, 8000, 9000, 9001, 9090, 9091, 19530)


def inspect_exposure(config: dict) -> dict:
    services = config.get("services", {})
    networks = config.get("networks", {})
    passed = set(services) == _SERVICES and networks.get("app", {}).get("internal") is True
    public = []
    for name, service in services.items():
        if service.get("network_mode") or service.get("privileged") or service.get("pid") == "host":
            passed = False
        ports = service.get("ports", [])
        if ports:
            public.append(name)
        if name == "caddy":
            passed &= ports in (["80:80", "443:443"], ["80:80", "443:443", "443:443/udp"])
        elif ports:
            passed = False
        allowed_networks = {"app", "egress"} if name == "backend" else {"app", "edge"} if name == "caddy" else {"app"}
        if set(service.get("networks", [])) != allowed_networks:
            passed = False
    return {
        "schema": "pilot_exposure/v1", "configuration_passed": bool(passed),
        "configuration_sha256": canonical_json_sha256(config),
        "public_services": sorted(name for name in public if name in _SERVICES),
        "live_exposure_verified": False,
    }


def plaintext_is_closed(status: int, location: str | None, hostname: str) -> bool:
    if status in {400, 403, 404, 421, 426}:
        return True
    if status not in {301, 302, 307, 308} or not location:
        return False
    try:
        parsed = urlsplit(location)
        return (
            parsed.scheme == "https" and parsed.hostname == hostname and parsed.port in (None, 443)
            and parsed.username is None and parsed.password is None
        )
    except ValueError:
        return False


def probe_exposure(hostname: str, *, all_tcp_ports: bool = False, tcp_workers: int = 4) -> dict:
    """Run only against an explicitly authorized deployment, without request content."""
    if type(tcp_workers) is not int or not 1 <= tcp_workers <= 64:
        raise ValueError("tcp_workers must be an integer from 1 through 64")
    result = {
        "http_closed": False, "https_health": False, "https_application": False,
        "tcp_probe_complete": False, "internal_ports_closed": False,
        "port_scope": "all_tcp" if all_tcp_ports else "declared_infrastructure_tcp",
    }
    try:
        with httpx.Client(timeout=5, follow_redirects=False, trust_env=False) as client:
            try:
                response = client.get(f"http://{hostname}/")
                result["http_closed"] = plaintext_is_closed(response.status_code, response.headers.get("location"), hostname)
            except httpx.ConnectError:
                # A missing HTTP listener is allowed only when TLS is proven below.
                result["http_closed"] = True
            health = client.get(f"https://{hostname}/api/health")
            result["https_health"] = health.status_code == 200 and health.json().get("data", {}).get("status") == "up"
            application = client.get(f"https://{hostname}/")
            result["https_application"] = application.status_code == 200 and 'id="app"' in application.text
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)})
        ports = tuple(range(1, 65536)) if all_tcp_ports else _PRIVATE_PORTS

        def closed(target: tuple[str, int]) -> bool | None:
            address, port = target
            try:
                with socket.create_connection((address, port), timeout=0.5):
                    return port in {22, 80, 443} if all_tcp_ports else False
            except ConnectionRefusedError:
                return True
            except OSError:
                return None

        with ThreadPoolExecutor(max_workers=tcp_workers) as pool:
            checks = list(pool.map(closed, ((address, port) for address in addresses for port in ports)))
        result["tcp_probe_complete"] = (
            bool(addresses) and len(checks) == len(addresses) * len(ports) and None not in checks
        )
        result["internal_ports_closed"] = bool(checks) and all(checks)
    except (httpx.HTTPError, OSError, ValueError):
        pass
    result["live_exposure_verified"] = all(result[key] for key in (
        "http_closed", "https_health", "https_application", "tcp_probe_complete", "internal_ports_closed",
    ))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Retain non-content Pilot exposure evidence")
    parser.add_argument("--compose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--authorized-host")
    parser.add_argument("--all-tcp-ports", action="store_true")
    parser.add_argument("--tcp-workers", type=int, choices=range(1, 65), default=4)
    args = parser.parse_args()
    if len(args.source_revision) != 40 or any(char not in "0123456789abcdef" for char in args.source_revision):
        parser.error("source revision must be an exact SHA-1 identity")
    if args.authorized_host and (
        any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for char in args.authorized_host)
        or "." not in args.authorized_host
    ):
        parser.error("authorized host must be a bare deployment DNS name")
    report = inspect_exposure(yaml.safe_load(args.compose.read_text()))
    if args.authorized_host:
        report.update(probe_exposure(
            args.authorized_host, all_tcp_ports=args.all_tcp_ports, tcp_workers=args.tcp_workers,
        ))
    report.update({"source_revision": args.source_revision, "checked_at": datetime.now(UTC).isoformat()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["configuration_passed"] and report["live_exposure_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
