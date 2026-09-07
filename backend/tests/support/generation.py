from app.extensions.provider_router import ApprovedGenerationRoute, ApprovedRouteProvider


def approved_test_route(*names: str) -> ApprovedGenerationRoute:
    """Explicit declaration for deterministic provider doubles, never live approval."""
    return ApprovedGenerationRoute(
        identity="provider_route:deterministic-fixture-v1",
        data_scope="team_shared_pilot",
        providers=tuple(ApprovedRouteProvider(
            f"configuration:{name}-test-v1", name, "deterministic-model", "team_shared_pilot", 30.0,
        ) for name in names),
        max_attempts=len(names),
        total_timeout_seconds=60.0,
    )


def generation_acceptance_payload(route: dict) -> dict:
    from tests.integration.test_delivery_acceptance import _record

    payload = _record()
    payload["conditions"]["generation_validation_mode"] = "local_development"
    identities = [route["route_identity"], *[item["approval_identity"] for item in route["providers"]]]
    payload["product_identities"].extend(identities)
    for name in ("answer-contract", "provider-route", "generation-provider", "generation-failure",
                 "generation-privacy", "generation-prompt-citation"):
        payload["checks"].append({
            "check_id": f"check:{name}", "result": "passed", "identity_dependencies": identities,
            "evidence_links": [f"evidence://ticket22/deterministic/{name}"],
        })
    return payload
