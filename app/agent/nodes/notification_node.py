"""Dispatches paper alerts to Discord (and Telegram if configured) before the approval interrupt."""

import logging
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from app.agent.state import AgentState
from app.core.config import settings
from app.models.schemas import AlertPayload, TechReleaseItem
from app.services.notifier import NotificationService

logger = logging.getLogger("techradar.notification_node")


async def notification_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Send alert notification with approval buttons before pausing at the human interrupt."""
    thread_id = "default-thread"
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id", thread_id)

    release_raw = state.get("release_item")
    if isinstance(release_raw, dict):
        release = TechReleaseItem(**release_raw)
    else:
        release = release_raw

    user_id = state.get("user_id", "default_user")
    discord_recipient = settings.DISCORD_USER_ID or settings.DISCORD_CHANNEL_ID or user_id

    # Look up user's discord_id if registered in db
    if user_id and user_id != "default_user":
        try:
            from app.core.database import AsyncSessionLocal
            from app.models.entities import UserProfile
            from sqlalchemy import select

            async with AsyncSessionLocal() as session:
                stmt = select(UserProfile).where(UserProfile.user_id == user_id)
                res = await session.execute(stmt)
                u = res.scalars().first()
                if u and u.discord_id:
                    discord_recipient = u.discord_id
        except Exception:
            pass

    payload = AlertPayload(
        thread_id=thread_id,
        release_title=release.title,
        release_summary=release.summary,
        source_url=release.source_url,
        matched_domains=state.get("matched_domains", []),
        relevance_score=state.get("relevance_score", 0.0),
        recipient_id=discord_recipient,
        channel="DISCORD",
    )

    notifier_service = NotificationService()
    success = await notifier_service.send_alert(payload)

    primary = getattr(settings, "PRIMARY_NOTIFICATION_CHANNEL", "DISCORD").upper()
    if primary == "BOTH":
        tg_recipient = settings.TELEGRAM_CHAT_ID or user_id
        if user_id and user_id != "default_user":
            try:
                from app.core.database import AsyncSessionLocal
                from app.models.entities import UserProfile
                from sqlalchemy import select

                async with AsyncSessionLocal() as session:
                    stmt = select(UserProfile).where(UserProfile.user_id == user_id)
                    res = await session.execute(stmt)
                    u = res.scalars().first()
                    if u and u.telegram_chat_id:
                        tg_recipient = u.telegram_chat_id
            except Exception:
                pass

        tg_payload = AlertPayload(
            thread_id=thread_id,
            release_title=release.title,
            release_summary=release.summary,
            source_url=release.source_url,
            matched_domains=state.get("matched_domains", []),
            relevance_score=state.get("relevance_score", 0.0),
            recipient_id=tg_recipient,
            channel="TELEGRAM",
        )
        await notifier_service.send_alert(tg_payload)

    if success:
        logger.info(f"Notification successfully posted to Discord for thread={thread_id}")
        return {
            "notification_status": "DISPATCHED_DISCORD",
        }
    else:
        err_msg = f"Failed to deliver Discord notification for thread={thread_id}"
        logger.error(err_msg)
        error_logs = list(state.get("error_logs", []))
        error_logs.append(err_msg)
        return {
            "notification_status": "DISPATCHED_DISCORD",
            "error_logs": error_logs,
        }
