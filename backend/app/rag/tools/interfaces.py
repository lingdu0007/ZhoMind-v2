from dataclasses import dataclass


@dataclass
class ToolSpec:
    name: str
    side_effect: str
    latency_tier: str
    stability_tier: str
    auth_scope: str
    input_schema_version: str
    output_schema_version: str
    cost_tier: str
