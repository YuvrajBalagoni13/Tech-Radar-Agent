"""
Telegram Timeout Escalation Node.
Triggered when an interactive Discord alert remains unacknowledged past the escalation SLA.
Dispatches high-priority Telegram alert with interactive inline action buttons (100% Free Forever).
"""

import logging
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from app.agent.state import AgentState
from app.models.schemas import AlertPayload, TechReleaseItem
from app.services.notifier import NotificationService

logger = logging.getLogger("techradar.escalation_node")


async def escalation_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Dispatches escalation notification to Telegram via Telegram Bot API.
    Interrupt checkpoint occurs immediately after this node.
    """
    thread_id = "default-thread"
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id", thread_id)

    release_raw = state.get("release_item")
    if isinstance(release_raw, dict):
        release = TechReleaseItem(**release_raw)
    else:
        release = release_raw

    payload = AlertPayload(
        thread_id=thread_id,
        release_title=release.title,
        release_summary=release.summary,
        source_url=release.source_url,
        matched_domains=state.get("matched_domains", []),
        relevance_score=state.get("relevance_score", 0.0),
        recipient_id=state.get("user_id", "default_user"),
        channel="TELEGRAM",
    )

    notifier_service = NotificationService()
    success = await notifier_service.send_alert(payload)

    if success:
        logger.info(f"Escalation alert delivered to Telegram for thread={thread_id}")
        return {
            "notification_status": "ESCALATED_TELEGRAM",
        }
    else:
        err_msg = f"Failed to deliver Telegram escalation for thread={thread_id}"
        logger.error(err_msg)
        error_logs = list(state.get("error_logs", []))
        error_logs.append(err_msg)
        return {
            "notification_status": "ESCALATED_TELEGRAM",
            "error_logs": error_logs,
        }
