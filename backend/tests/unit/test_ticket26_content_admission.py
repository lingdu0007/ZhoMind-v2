import pytest

from app.editorial_authority.schemas import editorial_secret_scan_findings, review_validation_reasons
from tests.unit.test_editorial_authority import _review_ready_entry


@pytest.mark.parametrize("material_class", [
    "secrets", "credentials", "personal_data", "customer_data", "raw_production_dump",
    "unredacted_logs", "unredacted_screenshots", "subset_authorized", "mixed_sensitivity",
    "dangerous_unpublished_exploit",
])
def test_review_rejects_content_outside_the_pilot_boundary(material_class):
    entry = _review_ready_entry()
    entry.sources[0]["content_admission"] = {
        "material_class": material_class, "audience": "all_admitted_members",
        "sensitivity": "restricted", "sanitized": True,
    }
    assert any(reason["code"] == "content_boundary_rejected" for reason in review_validation_reasons(entry))


def test_review_requires_explicit_all_member_admission():
    entry = _review_ready_entry()
    entry.sources[0].pop("content_admission", None)
    assert any(reason["code"] == "content_boundary_rejected" for reason in review_validation_reasons(entry))


@pytest.mark.parametrize("value", [
    "email: private.person@example.com",
    "ssn: 123-45-6789",
    "password=fixture-secret-value",
])
def test_recognizable_private_material_is_rejected_before_retention(value):
    assert editorial_secret_scan_findings({"body": {"decision_query": value}})


def test_public_locator_removes_credentials_and_unsafe_url_data():
    from app.documents.source_urls import sanitize_public_source_url

    raw = "https://reader:private-value@example.com/docs?version=1&api_key=private-value&next=https://internal.local#private"
    assert sanitize_public_source_url(raw) == "https://example.com/docs?version=1"


@pytest.mark.parametrize("query", [
    "access_key=fixture-private-value", "endpoint=https%3A%2F%2F10.1.2.3%2F",
    "version=10.1.2.3", "version=db.internal", "version=sk-" + "A" * 32,
    "version=0x7f.0.0.1", "version=0177.0.0.1", "version=2130706433",
    "revision=localhost.", "revision=db.internal.", "revision=metadata.google.internal.",
    "revision=127.0.0.1.",
    "revision=github_pat_" + "A" * 24,
])
def test_public_locator_removes_unapproved_or_private_query_values(query):
    from app.documents.source_urls import is_canonical_public_source_url, sanitize_public_source_url

    raw = f"https://example.com/docs?{query}"
    assert sanitize_public_source_url(raw) == "https://example.com/docs"
    assert is_canonical_public_source_url(raw) is False
    entry = _review_ready_entry()
    entry.sources[0]["public_url"] = raw
    assert any(reason["field"].endswith("public_url") for reason in review_validation_reasons(entry))


@pytest.mark.parametrize("url", [
    "https://10.1.2.3/doc", "https://127.0.0.1/doc", "https://[::1]/doc",
    "https://metadata.google.internal/doc", "https://localhost./doc",
    "https://example.com/%0asecret", "https://example.com/docs?X-Amz-Credential=private",
    "https://0x7f.0.0.1/doc", "https://0177.0.0.1/doc", "https://%30x7f.0.0.1/doc",
])
def test_source_authority_rejects_private_or_uncanonical_locations(url):
    entry = _review_ready_entry()
    entry.sources[0]["public_url"] = url
    assert any(reason["field"].endswith("public_url") for reason in review_validation_reasons(entry))


@pytest.mark.parametrize("host", [
    "0x7f.0.0.1", "0177.0.0.1", "%30x7f.0.0.1", "[2001:4860:4860::8888%25eth0]", "224.0.0.1",
])
def test_public_locator_rejects_non_public_or_noncanonical_hosts(host):
    from app.documents.source_urls import sanitize_public_source_url

    with pytest.raises(ValueError):
        sanitize_public_source_url(f"https://{host}/doc")


@pytest.mark.parametrize("sanitized", [1, False, "true", None])
def test_admission_requires_literal_sanitization_attestation(sanitized):
    entry = _review_ready_entry()
    entry.sources[0]["content_admission"]["sanitized"] = sanitized
    assert any(reason["code"] == "content_boundary_rejected" for reason in review_validation_reasons(entry))


@pytest.mark.parametrize("prefix", [
    "sk-", "%73%6b%2d", "%2573%256b%252d", "%252573%25256b%25252d",
])
def test_public_locator_rejects_recognizable_credentials_in_encoded_paths(prefix):
    from app.documents.source_urls import is_canonical_public_source_url, sanitize_public_source_url

    raw = "https://example.com/docs/" + prefix + "A" * 32
    with pytest.raises(ValueError, match="public source URL"):
        sanitize_public_source_url(raw)
    assert is_canonical_public_source_url(raw) is False
    entry = _review_ready_entry()
    entry.sources[0]["public_url"] = raw
    assert any(reason["field"].endswith("public_url") for reason in review_validation_reasons(entry))
    assert editorial_secret_scan_findings(entry.model_dump(mode="json"))
