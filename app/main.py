"""
FastAPI Enterprise Asynchronous Microservice.
Provides Discord interaction callbacks, Telegram bot webhooks,
REST APIs for developer profile configuration, and background periodic ingestion polling.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import get_radar_graph, resume_graph
from app.agent.state import AgentState
from app.bot.discord_client import (
    DiscordInteractionHandler,
    start_discord_gateway,
    stop_discord_gateway,
    sync_user_embedding,
)
from app.bot.telegram_handler import (
    TelegramWebhookHandler,
    start_telegram_polling,
    stop_telegram_polling,
)
from app.core.config import settings
from app.core.database import close_db, get_db, init_db
from app.models.entities import UserProfile
from app.models.schemas import (
    InteractionCallback,
    TechReleaseItem,
    TelegramUpdate,
    UserProfileCreate,
    UserProfileResponse,
    UserProfileUpdate,
)
from app.services.ingestion import IngestionService

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("techradar.main")

_polling_task: Optional[asyncio.Task] = None


async def periodic_ingestion_worker() -> None:
    """
    Background worker periodically polling HackerNews, arXiv, and GitHub feeds.
    Dispatches newly discovered items through the LangGraph StateMachine.
    """
    logger.info("Starting background periodic ingestion worker...")
    ingestion_service = IngestionService()
    graph = get_radar_graph()

    while True:
        try:
            logger.info("Executing scheduled feed poll cycle...")
            sources = None
            limit = None
            plan = None
            try:
                from app.bot.telegram_handler import _in_memory_profiles
                from app.agent.nodes.planner_node import generate_search_plan
                if _in_memory_profiles:
                    u = next(iter(_in_memory_profiles.values()))
                    sources = getattr(u, "enabled_sources", None)
                    limit = getattr(u, "scan_limit", None)
                    plan = await generate_search_plan(u, discovery_mode="latest")
            except Exception:
                pass

            items = await ingestion_service.ingest_all(sources=sources, limit_per_source=limit, plan=plan)
            for item in items:
                thread_id = str(uuid.uuid4())
                initial_state: AgentState = {
                    "release_item": item,
                    "user_id": "default_user",
                    "search_plan": plan.model_dump() if plan else None,
                    "relevance_score": 0.0,
                    "is_relevant": False,
                    "matched_domains": [],
                    "filter_reason": "Pending evaluation",
                    "notification_status": "INITIALIZED",
                    "user_prompt_override": None,
                    "research_notes": [],
                    "tutorial_markdown": None,
                    "pdf_artifact_path": None,
                    "error_logs": [],
                }
                config = {"configurable": {"thread_id": thread_id}}
                # Fire and forget execution for each item
                asyncio.create_task(graph.ainvoke(initial_state, config=config))

        except asyncio.CancelledError:
            logger.info("Periodic ingestion worker received cancellation.")
            break
        except Exception as e:
            logger.error(f"Error in periodic ingestion worker: {e}", exc_info=True)

        interval_seconds = max(settings.INGESTION_INTERVAL_MINUTES * 60, 60)
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager orchestrating startup and shutdown procedures.
    Initializes PostgreSQL extensions, tables, and starts background polling tasks.
    """
    global _polling_task
    logger.info("Starting Autonomous Tech Radar & Just-In-Time Learning Agent...")

    # 1. Initialize PostgreSQL schemas and extensions
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database bootstrap warning (proceeding in degraded mode): {e}")

    # 2. Warm up LangGraph StateMachine
    _ = get_radar_graph()

    # 3. Start background ingestion worker
    _polling_task = asyncio.create_task(periodic_ingestion_worker())

    # 4. Start Discord Gateway Bot for Direct Messages (if token configured)
    await start_discord_gateway()

    # 5. Start Telegram Long-Polling Worker (if token configured)
    await start_telegram_polling()

    yield

    # Teardown logic
    logger.info("Shutting down Tech Radar services...")
    await stop_telegram_polling()
    await stop_discord_gateway()

    if _polling_task and not _polling_task.done():
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass

    await close_db()
    logger.info("Application shutdown completed.")


app = FastAPI(
    title="Autonomous Tech Radar & Just-In-Time Learning Agent",
    description=(
        "Enterprise-grade LangGraph state machine with pgvector cosine filtering, "
        "interactive Discord / Telegram notifications, and WeasyPrint PDF synthesis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint confirming service status and operational parameters."""
    return {
        "status": "healthy",
        "service": "tech-radar-agent",
        "environment": settings.ENVIRONMENT,
        "default_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD,
        "escalation_sla_hours": settings.ESCALATION_TIMEOUT_HOURS,
    }


# -----------------------------------------------------------------------------
# Bot Webhooks & Interaction Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/v1/bot/discord/interactions", tags=["Discord Bot"])
async def discord_interactions(request: Request) -> Dict[str, Any]:
    """
    Discord Gateway interaction receiver handling slash commands and button clicks.
    """
    payload = await request.json()
    interaction_type = payload.get("type")

    # Discord PING handshake (Type 1)
    if interaction_type == 1:
        return {"type": 1}

    # Slash Command (Type 2)
    elif interaction_type == 2:
        data = payload.get("data", {})
        cmd_name = data.get("name")
        options_list = data.get("options", [])
        options = {opt.get("name"): opt.get("value") for opt in options_list}
        user_info = payload.get("member", {}).get("user") or payload.get("user", {})
        discord_id = user_info.get("id", "anonymous")

        return await DiscordInteractionHandler.handle_slash_command(
            name=cmd_name,
            options=options,
            user_discord_id=discord_id,
        )

    # Message Component / Button Click (Type 3)
    elif interaction_type == 3:
        data = payload.get("data", {})
        custom_id = data.get("custom_id", "")
        user_info = payload.get("member", {}).get("user") or payload.get("user", {})
        discord_id = user_info.get("id", "anonymous")

        return await DiscordInteractionHandler.handle_button_click(
            custom_id=custom_id,
            user_discord_id=discord_id,
        )

    return {"type": 4, "data": {"content": "Unhandled interaction."}}


@app.post("/api/v1/bot/telegram/webhook", tags=["Telegram Bot"])
async def telegram_webhook(update: TelegramUpdate) -> Dict[str, Any]:
    """
    Telegram Bot API Webhook.
    Receives text commands (/start, /track, /ignore, /status, /threshold, /yes, /skip)
    and inline keyboard callback clicks (1-click tutorial synthesis or dismissal).
    """
    return await TelegramWebhookHandler.process_update(update)


# -----------------------------------------------------------------------------
# User Profile Management Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/v1/users", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED, tags=["User Profiles"])
async def create_user_profile(payload: UserProfileCreate, db: AsyncSession = Depends(get_db)):
    """Register new developer profile and compute initial vector embedding."""
    stmt = select(UserProfile).where(UserProfile.user_id == payload.user_id)
    existing = (await db.execute(stmt)).scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail=f"User '{payload.user_id}' already registered.")

    user = UserProfile(
        user_id=payload.user_id,
        discord_id=payload.discord_id,
        telegram_chat_id=payload.telegram_chat_id,
        tracked_domains=payload.tracked_domains,
        ignored_keywords=payload.ignored_keywords,
        profile_summary=payload.profile_summary or "",
        similarity_threshold=payload.similarity_threshold or 0.82,
    )
    db.add(user)
    await db.flush()
    await sync_user_embedding(user)
    await db.commit()
    await db.refresh(user)
    return user


@app.get("/api/v1/users/{user_id}", response_model=UserProfileResponse, tags=["User Profiles"])
async def get_user_profile(user_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch developer profile and vector embedding status."""
    stmt = select(UserProfile).where(UserProfile.user_id == user_id)
    user = (await db.execute(stmt)).scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User profile not found.")
    return user


@app.put("/api/v1/users/{user_id}/domains", response_model=UserProfileResponse, tags=["User Profiles"])
async def update_user_domains(user_id: str, payload: UserProfileUpdate, db: AsyncSession = Depends(get_db)):
    """Update tracked domains or negative keywords with dynamic vector recalculation."""
    stmt = select(UserProfile).where(UserProfile.user_id == user_id)
    user = (await db.execute(stmt)).scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User profile not found.")

    if payload.tracked_domains is not None:
        user.tracked_domains = payload.tracked_domains
    if payload.ignored_keywords is not None:
        user.ignored_keywords = payload.ignored_keywords
    if payload.similarity_threshold is not None:
        user.similarity_threshold = payload.similarity_threshold
    if payload.profile_summary is not None:
        user.profile_summary = payload.profile_summary
    if payload.discord_id is not None:
        user.discord_id = payload.discord_id
    if payload.telegram_chat_id is not None:
        user.telegram_chat_id = payload.telegram_chat_id

    await sync_user_embedding(user)
    await db.commit()
    await db.refresh(user)
    return user


# -----------------------------------------------------------------------------
# Radar Ingestion & Thread Control Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/v1/radar/ingest", tags=["Radar Execution"])
async def trigger_ingestion(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Manually trigger immediate multi-source feed ingestion cycle."""
    async def _run():
        service = IngestionService()
        items = await service.ingest_all()
        graph = get_radar_graph()
        for it in items:
            t_id = str(uuid.uuid4())
            state: AgentState = {
                "release_item": it,
                "user_id": "default_user",
                "relevance_score": 0.0,
                "is_relevant": False,
                "matched_domains": [],
                "filter_reason": "Pending evaluation",
                "notification_status": "INITIALIZED",
                "user_prompt_override": None,
                "research_notes": [],
                "tutorial_markdown": None,
                "pdf_artifact_path": None,
                "error_logs": [],
            }
            asyncio.create_task(graph.ainvoke(state, config={"configurable": {"thread_id": t_id}}))

    background_tasks.add_task(_run)
    return {"message": "Ingestion job dispatched asynchronously."}


@app.post("/api/v1/radar/threads/{thread_id}/resume", tags=["Radar Execution"])
async def resume_workflow_thread(thread_id: str, callback: InteractionCallback) -> Dict[str, Any]:
    """Manually resume an interrupted LangGraph checkpoint thread via REST."""
    result = await resume_graph(
        thread_id=thread_id,
        action=callback.action,
        user_prompt_override=callback.override_prompt,
    )
    return {
        "thread_id": thread_id,
        "status": result.get("notification_status"),
        "pdf_artifact": result.get("pdf_artifact_path"),
    }
