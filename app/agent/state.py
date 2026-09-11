"""
LangGraph AgentState schema definition.
Implements immutable state transitions across multi-stage filtering, alerting,
human-in-the-loop callbacks, and artifact generation.
"""

from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict
from app.models.schemas import TechReleaseItem


class AgentState(TypedDict):
    """
    Core state schema passed through the LangGraph StateGraph nodes.
    Maintains checkpointed context across asynchronous human-in-the-loop interactions.
    """

    release_item: Union[TechReleaseItem, Dict[str, Any]]
    user_id: str
    relevance_score: float
    is_relevant: bool
    matched_domains: List[str]
    filter_reason: str
    notification_status: str  # DISPATCHED_DISCORD, ESCALATED_TELEGRAM, USER_APPROVED, USER_IGNORED
    user_prompt_override: Optional[str]
    research_notes: List[Dict[str, Any]]
    tutorial_markdown: Optional[str]
    pdf_artifact_path: Optional[str]
    search_plan: Optional[Dict[str, Any]]
    error_logs: List[str]
