import os
import platform
import re
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.operations.chat_capacity import get_chat_admission_gate
from app.operations.limits import MAX_DOCUMENT_BUILD_WORKERS
from app.retrieval.candidate_pool import AuthorizedRetrievalCandidatePool
from app.retrieval.policy import get_retrieval_policy
from app.settings.generation_routes import GenerationRouteService
from app.settings.runtime import get_runtime_settings


async def measurement_snapshot(session: AsyncSession) -> dict:
    settings = get_runtime_settings()
    policy = get_retrieval_policy(settings)
    corpus = await AuthorizedRetrievalCandidatePool(session, settings=settings).measurement_snapshot()
    route = (await GenerationRouteService(session).read())["active"]
    vcpu = os.cpu_count() or 1
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    host = {"node": platform.node(), "architecture": platform.machine(), "vcpu": vcpu, "memory_mib": memory}
    revision = settings.generation_product_revision
    return {
        **corpus,
        "deployment": "deployment:" + canonical_json_sha256(settings.generation_deployment_identity),
        "host": "host:" + canonical_json_sha256(host), "host_vcpu": vcpu, "host_memory_mib": memory,
        "configuration": get_chat_admission_gate().configuration()["identity"],
        "provider_route": route["route_identity"] if route else None,
        "retrieval_profile": "retrieval_profile:" + canonical_json_sha256(asdict(policy)),
        "chunk_envelope": "configuration:" + canonical_json_sha256({
            "items": policy.evidence_max_items, "per_snapshot": policy.evidence_max_chars_per_snapshot,
            "total_chars": policy.evidence_max_total_chars,
        }),
        "product_revision": revision if re.fullmatch(r"product_revision:[0-9a-f]{40}", revision) else None,
        "background_workers": MAX_DOCUMENT_BUILD_WORKERS,
    }
