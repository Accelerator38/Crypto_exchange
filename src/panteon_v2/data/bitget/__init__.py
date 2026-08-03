"""Bitget public market data layer."""

from .profile import (
    BITGET_DATA_PROFILE_SCHEMA_VERSION,
    BitgetDataProfile,
    BitgetDataProfileError,
    load_bitget_data_profile,
)
from .segment_store import (
    BITGET_SEGMENT_MANIFEST_SCHEMA_VERSION,
    BitgetDataCollectorLock,
    BitgetSegmentStore,
    RecoverySummary,
    SegmentValidationError,
    recover_unsealed_sessions,
    validate_data_session,
)
from .rest_reconciler import BitgetPublicRestReconciler, RestEvent
from .materializer import (
    BITGET_FRAME_SCHEMA_VERSION,
    BITGET_FRAME_SCHEMA_VERSION_V2,
    FrameDatasetSummary,
    materialize_frame_1s,
    materialize_frame_1s_v2,
    validate_frame_dataset,
    validate_frame_dataset_v2,
)
from .microstructure_baseline import (
    BASELINE_SCHEMA_VERSION,
    evaluate_fixed_baselines,
    render_baseline_markdown,
)
from .event_study import (
    EVENT_STUDY_SCHEMA_VERSION,
    evaluate_event_study,
    render_event_study_markdown,
)
from .maker_event_study import (
    MAKER_EVENT_STUDY_SCHEMA_VERSION,
    evaluate_maker_event_study,
    render_maker_event_study_markdown,
)
from .precursor_campaign import (
    CAMPAIGN_SCHEMA_VERSION as PRECURSOR_CAMPAIGN_SCHEMA_VERSION,
    run_precursor_campaign,
    run_precursor_campaign_v2,
    run_precursor_campaign_v3,
    render_precursor_campaign_markdown,
)
from .evidence_session import (
    EVIDENCE_SESSION_SCHEMA_VERSION,
    run_precursor_evidence_session,
)
from .evidence_session_v2 import (
    EVIDENCE_SESSION_SCHEMA_VERSION_V2,
    run_precursor_evidence_session_v2,
)
from .evidence_session_v3 import (
    EVIDENCE_SESSION_SCHEMA_VERSION_V3,
    run_precursor_evidence_session_v3,
)
from .research_dataset import (
    RESEARCH_DATASET_SCHEMA_VERSION,
    RESEARCH_SCHEMA_VERSION,
    ResearchDatasetSummary,
    materialize_research_dataset,
    validate_research_dataset,
)
from .research_dataset_v2 import (
    RESEARCH_DATASET_SCHEMA_VERSION_V2,
    RESEARCH_SCHEMA_VERSION_V2,
    materialize_research_dataset_v2,
    materialize_research_dataset_v3,
    validate_research_dataset_v2,
    validate_research_dataset_v3,
)
from .websocket_collector import (
    BitgetPublicMessageDecoder,
    DecodedEvent,
    build_subscription_request,
    run_public_collection,
)

__all__ = [
    "BITGET_DATA_PROFILE_SCHEMA_VERSION",
    "BITGET_FRAME_SCHEMA_VERSION",
    "BITGET_FRAME_SCHEMA_VERSION_V2",
    "BITGET_SEGMENT_MANIFEST_SCHEMA_VERSION",
    "BASELINE_SCHEMA_VERSION",
    "EVENT_STUDY_SCHEMA_VERSION",
    "MAKER_EVENT_STUDY_SCHEMA_VERSION",
    "PRECURSOR_CAMPAIGN_SCHEMA_VERSION",
    "EVIDENCE_SESSION_SCHEMA_VERSION",
    "EVIDENCE_SESSION_SCHEMA_VERSION_V2",
    "EVIDENCE_SESSION_SCHEMA_VERSION_V3",
    "RESEARCH_DATASET_SCHEMA_VERSION",
    "RESEARCH_SCHEMA_VERSION",
    "RESEARCH_SCHEMA_VERSION_V2",
    "RESEARCH_DATASET_SCHEMA_VERSION_V2",
    "BitgetDataProfile",
    "BitgetDataProfileError",
    "BitgetDataCollectorLock",
    "BitgetPublicMessageDecoder",
    "BitgetPublicRestReconciler",
    "BitgetSegmentStore",
    "DecodedEvent",
    "FrameDatasetSummary",
    "ResearchDatasetSummary",
    "RecoverySummary",
    "RestEvent",
    "SegmentValidationError",
    "build_subscription_request",
    "load_bitget_data_profile",
    "materialize_frame_1s",
    "materialize_frame_1s_v2",
    "materialize_research_dataset",
    "materialize_research_dataset_v2",
    "materialize_research_dataset_v3",
    "evaluate_fixed_baselines",
    "evaluate_event_study",
    "evaluate_maker_event_study",
    "recover_unsealed_sessions",
    "run_public_collection",
    "render_baseline_markdown",
    "render_event_study_markdown",
    "render_maker_event_study_markdown",
    "render_precursor_campaign_markdown",
    "run_precursor_campaign",
    "run_precursor_campaign_v2",
    "run_precursor_campaign_v3",
    "run_precursor_evidence_session",
    "run_precursor_evidence_session_v2",
    "run_precursor_evidence_session_v3",
    "validate_data_session",
    "validate_frame_dataset",
    "validate_frame_dataset_v2",
    "validate_research_dataset",
    "validate_research_dataset_v2",
    "validate_research_dataset_v3",
]
