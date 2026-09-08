from __future__ import annotations

import re
from collections.abc import Mapping

HIGH_IMPACT_CLAIM_KINDS = frozenset({"prescriptive", "numeric", "version", "security", "privacy", "high_impact"})

_HIGH_IMPACT_TEXT = re.compile(
    r"""
    \b(?:must|shall|require|requires|never|only)\b
    |\b(?:security|privacy|authentication|authorization|permissions?|prompt[- ]?injection)\b
    |\b(?:version|v\d+(?:\.\d+)+)\b
    |(?:<=|>=|==|!=|&&|\|\|)
    |\b\d+(?:\.\d+)?\s*(?:%|ms|seconds?|minutes?|hours?|days?)\b
    |(?:\u5fc5\u987b|\u5e94\u5f53|\u9700\u8981|\u4e0d\u5f97|\u7981\u6b62|\u4ec5\u80fd|\u4ec5\u53ef)
    |(?:\u5b89\u5168|\u9690\u79c1|\u8ba4\u8bc1|\u6388\u6743|\u6743\u9650|\u63d0\u793a\u6ce8\u5165)
    |(?:\u7248\u672c|\u53d1\u5e03|\u751f\u4ea7\u73af\u5883|\u590d\u6838)
    |\d+(?:\.\d+)?\s*(?:%|\u6beb\u79d2|\u79d2|\u5206\u949f|\u5c0f\u65f6|\u5929)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def content_indicates_high_impact(value: object) -> bool:
    return isinstance(value, str) and bool(_HIGH_IMPACT_TEXT.search(value))


def is_high_impact_claim(claim: Mapping[str, object]) -> bool:
    return claim.get("claim_kind") in HIGH_IMPACT_CLAIM_KINDS or content_indicates_high_impact(claim.get("statement"))


def is_material_claim(claim: Mapping[str, object]) -> bool:
    return claim.get("material") is True or is_high_impact_claim(claim)
