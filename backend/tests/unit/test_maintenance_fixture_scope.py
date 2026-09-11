from copy import deepcopy

import pytest

from app.common.exceptions import AppError
from app.maintenance.fixtures import qualify_fixture_scope


@pytest.mark.parametrize("corruption", ["missing", "empty", "foreign_scope", "foreign_review", "foreign_reference", "private_field"])
def test_fixture_scope_requires_a_retained_nonpersonal_target(corruption):
    scope = {
        "entry_versions": [{
            "entry_identity": "entry:reported-entry",
            "publication_identity": "published_knowledge_version:original",
            "revision_identity": "editorial_revision:reported-entry.r1",
        }],
        "gap_contexts": [],
    }
    item = {"affected_scope": scope}
    fixture = {
        "affected_scope": deepcopy(scope),
        "publication_review": {"entry_identity": "entry:reported-entry"},
        "reference_publication_reviews": [{"entry_identity": "entry:reported-entry"}],
    }
    qualify_fixture_scope(item, fixture)
    if corruption == "missing":
        del fixture["affected_scope"]
    elif corruption == "empty":
        fixture["affected_scope"] = {"entry_versions": [], "gap_contexts": []}
    elif corruption == "foreign_scope":
        fixture["affected_scope"]["entry_versions"][0]["entry_identity"] = "entry:foreign-entry"
    elif corruption == "foreign_review":
        fixture["publication_review"]["entry_identity"] = "entry:foreign-entry"
    elif corruption == "foreign_reference":
        fixture["reference_publication_reviews"][0]["entry_identity"] = "entry:foreign-entry"
    else:
        fixture["affected_scope"]["question"] = "private question must not become scope"
    with pytest.raises(AppError) as rejected:
        qualify_fixture_scope(item, fixture)
    assert rejected.value.status_code == 409
    assert rejected.value.code == "MAINTENANCE_EVIDENCE_REQUIRED"
