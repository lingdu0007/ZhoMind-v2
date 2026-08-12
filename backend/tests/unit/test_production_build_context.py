from pathlib import Path

import yaml


def test_production_build_context_excludes_host_dependencies_and_test_artifacts() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    patterns = {
        line.strip()
        for line in (repository_root / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "frontend/node_modules" in patterns
    assert "frontend/test-artifacts" in patterns
    assert "backend/.venv" in patterns
    assert ".git" in patterns


def test_production_minio_healthcheck_uses_the_client_shipped_in_the_image() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    compose = yaml.safe_load((repository_root / "deploy" / "production" / "compose.yml").read_text(encoding="utf-8"))

    assert compose["services"]["minio"]["healthcheck"]["test"] == ["CMD", "mc", "ready", "local"]
