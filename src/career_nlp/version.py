"""Single source of truth for versioning.

ASSET_VERSION  - semantic version of the component (code + behaviour).
SCHEMA_VERSION - version of the structured input/output contract. Bumped only
                 when a response field is removed/renamed/re-typed; adding
                 optional fields does not change it.
"""

ASSET_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
CONFIDENCE_METHOD = "evidence_heuristic_v1"
