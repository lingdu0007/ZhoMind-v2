from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "scripts" / "validate-evidence-bundle.py"
EXAMPLE = REPO_ROOT / "public-evidence" / "example"


def _validate(bundle_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(bundle_dir)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_accepted_candidate_requires_single_and_five_concurrent_profiles(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "accepted-candidate"
    shutil.copytree(EXAMPLE, bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_candidate"]["status"] = "accepted"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = _validate(bundle_dir)

    assert result.returncode == 1
    assert "release-completeness" in result.stdout
    assert "concurrency" in result.stdout
