"""Compiles markdown into PDF and sends it via Discord or Telegram."""

import logging
from typing import Any, Dict
from app.agent.state import AgentState
from app.core.config import settings
from app.models.schemas import TechReleaseItem
from app.services.notifier import NotificationService
from app.services.pdf_generator import PDFGeneratorService

logger = logging.getLogger("techradar.compiler_node")


async def compiler_node(state: AgentState) -> Dict[str, Any]:
    """Render markdown brief as PDF and send to the user's notification channel."""
    release_raw = state.get("release_item")
    if isinstance(release_raw, dict):
        release = TechReleaseItem(**release_raw)
    else:
        release = release_raw

    markdown_text = state.get("tutorial_markdown", "")
    title = f"Technical Brief: {release.title}"

    pdf_service = PDFGeneratorService()
    logger.info(f"Initiating PDF compilation for technical brief '{title}'...")

    pdf_path = await pdf_service.generate_pdf(
        markdown_text=markdown_text,
        title=title,
        filename_prefix=release.title[:20],
    )
    logger.info(f"PDF successfully generated at: {pdf_path}")

    # Dispatch artifact to Discord (User DM or Channel)
    notifier_service = NotificationService()
    caption = f"Summarized technical explanation & brief compiled for **{release.title}**."
    user_id = state.get("user_id", "default_user")

    discord_recipient = settings.DISCORD_USER_ID or settings.DISCORD_CHANNEL_ID or user_id
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

    primary = getattr(settings, "PRIMARY_NOTIFICATION_CHANNEL", "DISCORD").upper()
    notification_status = state.get("notification_status")

    # Dispatch to Discord if primary is DISCORD or BOTH, or if user interacted via Discord
    if primary in ("DISCORD", "BOTH") or notification_status in ("DISPATCHED_DISCORD", "USER_APPROVED"):
        await notifier_service.send_artifact(
            channel="DISCORD",
            recipient=discord_recipient,
            file_path=pdf_path,
            caption=caption,
        )

    # Dispatch to Telegram if primary is TELEGRAM or BOTH, or if flow was escalated to Telegram
    if primary in ("TELEGRAM", "BOTH") or notification_status == "ESCALATED_TELEGRAM":
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

        await notifier_service.send_artifact(
            channel="TELEGRAM",
            recipient=tg_recipient,
            file_path=pdf_path,
            caption=caption,
        )

    return {
        "pdf_artifact_path": pdf_path,
    }
