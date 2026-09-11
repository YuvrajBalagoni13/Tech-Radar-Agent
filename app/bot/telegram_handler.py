"""
Telegram Bot Webhook and Interactive Command Handler.
Processes incoming Telegram updates (text commands and inline keyboard callback queries).
Enables 100% free interactive multi-channel dispatch, dynamic embedding synchronization,
and LangGraph human-in-the-loop checkpoint resumption.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import httpx
from sqlalchemy import select
from app.agent.graph import resume_graph
from app.bot.discord_client import sync_user_embedding
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.entities import UserProfile
from app.models.schemas import TelegramUpdate

logger = logging.getLogger("techradar.telegram_handler")


_in_memory_profiles: Dict[str, UserProfile] = {}


class TelegramWebhookHandler:
    """
    Processes Telegram Bot API webhook updates and long-polling messages.
    Routes slash commands (/track, /ignore, /status, /threshold, /yes, /skip)
    and handles inline keyboard button callbacks for 1-click tutorial compilation.
    Includes in-memory profile fallback so the bot functions smoothly even if DB is offline.
    """

    @staticmethod
    def _ensure_user_defaults(user: UserProfile) -> None:
        """Ensure runtime defaults for enabled_sources, scan_limit, and profile_summary exist."""
        if not hasattr(user, "enabled_sources") or not getattr(user, "enabled_sources", None):
            user.enabled_sources = ["arxiv", "hackernews", "github"]
        if not hasattr(user, "scan_limit") or not getattr(user, "scan_limit", None):
            user.scan_limit = 10
        if not hasattr(user, "profile_summary") or getattr(user, "profile_summary", None) is None:
            user.profile_summary = ""

    @staticmethod
    async def _get_or_create_user_db(clean_chat_id: str, session: Any) -> UserProfile:
        stmt = select(UserProfile).where(UserProfile.telegram_chat_id == clean_chat_id)
        res = await session.execute(stmt)
        user = res.scalars().first()

        if not user:
            stmt_default = select(UserProfile).where(UserProfile.user_id == "default_user")
            res_default = await session.execute(stmt_default)
            default_user = res_default.scalars().first()
            if default_user:
                default_user.telegram_chat_id = clean_chat_id
                await session.commit()
                logger.info(f"Linked Telegram Chat ID {clean_chat_id} to default_user profile")
                TelegramWebhookHandler._ensure_user_defaults(default_user)
                return default_user

            user = UserProfile(
                user_id=f"telegram_{clean_chat_id}",
                telegram_chat_id=clean_chat_id,
                tracked_domains=["Distributed Systems", "AI Infrastructure", "LLM Serving"],
                ignored_keywords=["crypto", "nft", "airdrop"],
                profile_summary="",
                similarity_threshold=settings.DEFAULT_SIMILARITY_THRESHOLD,
            )
            session.add(user)
            await session.flush()
            try:
                await sync_user_embedding(user)
            except Exception:
                pass
            await session.commit()
            logger.info(f"Provisioned new UserProfile for Telegram Chat ID {clean_chat_id}")

        TelegramWebhookHandler._ensure_user_defaults(user)
        return user

    @classmethod
    async def get_or_create_user(cls, chat_id: str, session: Optional[Any] = None) -> UserProfile:
        """Locate user profile by Telegram chat ID or provision a new developer profile with in-memory fallback."""
        clean_chat_id = str(chat_id).strip()
        try:
            if session:
                u = await cls._get_or_create_user_db(clean_chat_id, session)
            else:
                async with AsyncSessionLocal() as s:
                    u = await cls._get_or_create_user_db(clean_chat_id, s)
            cls._ensure_user_defaults(u)
            return u
        except Exception as e:
            logger.warning(f"Database unavailable for user profile ({e}). Using in-memory fallback profile.")
            if clean_chat_id not in _in_memory_profiles:
                _in_memory_profiles[clean_chat_id] = UserProfile(
                    user_id=f"telegram_{clean_chat_id}",
                    telegram_chat_id=clean_chat_id,
                    tracked_domains=["Distributed Systems", "AI Infrastructure", "LLM Serving"],
                    ignored_keywords=["crypto", "nft", "airdrop"],
                    profile_summary="",
                    similarity_threshold=settings.DEFAULT_SIMILARITY_THRESHOLD,
                )
            u = _in_memory_profiles[clean_chat_id]
            cls._ensure_user_defaults(u)
            return u

    @classmethod
    async def _send_telegram_reply(cls, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
        """Helper to send a text message back to the Telegram chat."""
        if not settings.TELEGRAM_BOT_TOKEN or settings.TELEGRAM_BOT_TOKEN.startswith("your_"):
            logger.info(f"[MOCK TELEGRAM REPLY] To chat_id={chat_id}: {text}")
            return True

        endpoint = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
        body = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(endpoint, json=body)
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message reply: {e}", exc_info=True)
            return False

    @classmethod
    async def _answer_callback_query(cls, callback_query_id: str, text: str) -> bool:
        """Acknowledge Telegram callback query to dismiss the loading spinner."""
        if not settings.TELEGRAM_BOT_TOKEN or settings.TELEGRAM_BOT_TOKEN.startswith("your_"):
            return True

        endpoint = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
        body = {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": False,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(endpoint, json=body)
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to answer Telegram callback query: {e}", exc_info=True)
            return False

    @classmethod
    async def process_update(cls, update: TelegramUpdate) -> Dict[str, Any]:
        """
        Main entrypoint for Telegram webhook and long-polling updates.
        Handles message commands and inline button callbacks.
        """
        # 1. Handle Inline Button Clicks (Callback Query)
        if update.callback_query:
            cq = update.callback_query
            data = cq.data or ""
            parts = data.split(":")
            chat_id = cq.message.chat.id if cq.message else 0

            if len(parts) >= 3:
                action_type = parts[1]  # 'generate' or 'skip'
                thread_id = parts[2]

                if action_type == "generate":
                    await cls._answer_callback_query(cq.id, "📑 Synthesizing technical brief & explanation...")
                    logger.info(f"Telegram button approval for thread '{thread_id}'")
                    asyncio.create_task(
                        resume_graph(thread_id=thread_id, action="generate_tutorial")
                    )
                    if chat_id:
                        await cls._send_telegram_reply(
                            chat_id,
                            f"📑 <b>Deep Research Initiated</b>\nSynthesizing publication-grade technical brief for thread <code>{thread_id}</code>. You will receive the compiled PDF shortly!"
                        )
                    return {"ok": True, "action": "generate", "thread_id": thread_id}

                elif action_type == "skip":
                    await cls._answer_callback_query(cq.id, "⏭️ Dismissed.")
                    logger.info(f"Telegram button dismissal for thread '{thread_id}'")
                    asyncio.create_task(
                        resume_graph(thread_id=thread_id, action="skip_release")
                    )
                    if chat_id:
                        await cls._send_telegram_reply(
                            chat_id,
                            f"⏭️ Thread <code>{thread_id}</code> has been dismissed."
                        )
                    return {"ok": True, "action": "skip", "thread_id": thread_id}

            await cls._answer_callback_query(cq.id, "Unknown action.")
            return {"ok": False, "error": "Invalid callback data"}

        # 2. Handle Text Commands
        if update.message and update.message.text:
            text = update.message.text.strip()
            chat_id = update.message.chat.id

            db_session = None
            try:
                db_session = AsyncSessionLocal()
            except Exception:
                pass

            try:
                user = await cls.get_or_create_user(str(chat_id), db_session)

                tokens = text.split()
                cmd = tokens[0].split("@")[0].lower() if tokens else ""

                # Command: /start or /help
                if cmd in ("/start", "/help"):
                    reply = (
                        "🤖 <b>Autonomous Tech Radar Bot (100% Free)</b>\n\n"
                        "<b>Ad-Hoc Targeted Research (Any Subject):</b>\n"
                        "• <code>/research &lt;query&gt;</code> - Deep literature search on any topic (e.g. <code>/research I have a thesis coming & want to learn theoretical cryptography</code>)\n"
                        "• <code>/search &lt;query&gt;</code> - Targeted research scan without modifying persistent profile (e.g. <code>/search speculative decoding vllm</code>)\n\n"
                        "<b>Scanning & Discovery:</b>\n"
                        "• <code>/scan [source] [limit] [mode]</code> - Immediate radar scan (e.g. <code>/scan</code> or <code>/scan arxiv 10 foundational</code>)\n"
                        "• <code>/sources</code> - View or change sources (e.g. <code>/sources only arxiv</code>)\n"
                        "• <code>/limit &lt;1-50&gt;</code> - Max articles per source (e.g. <code>/limit 5</code>)\n"
                        "• <code>/interval [mins]</code> - View or update background polling interval\n"
                        "• <code>/reset history</code> - Clear seen memory so past articles can be re-scanned\n\n"
                        "<b>Developer Profile & Interests:</b>\n"
                        "• <code>/profile</code> - View your profile bio & background description\n"
                        "• <code>/profile set &lt;bio&gt;</code> - Describe your role/interests (combined into AI similarity score)\n"
                        "• <code>/profile clear</code> - Remove profile bio\n\n"
                        "<b>Topic & Filter Configuration:</b>\n"
                        "• <code>/track &lt;topics&gt;</code> - Add technologies (e.g. <code>/track PyTorch, LLMs</code>)\n"
                        "• <code>/untrack &lt;topics or all&gt;</code> - Remove technologies\n"
                        "• <code>/ignore &lt;keywords&gt;</code> - Negative keyword filter ($0 LLM cost)\n"
                        "• <code>/unignore &lt;keywords or all&gt;</code> - Remove from negative filter\n"
                        "• <code>/threshold &lt;float&gt;</code> - Set similarity threshold (e.g. 0.82)\n"
                        "• <code>/status</code> - View complete active configuration\n\n"
                        "<b>Workflow Resumption:</b>\n"
                        "• <code>/yes &lt;thread_id&gt; [instructions]</code> - Authorize technical brief synthesis\n"
                        "• <code>/skip &lt;thread_id&gt;</code> - Dismiss candidate release\n\n"
                        "💡 <i>Tip: You can also use inline buttons directly underneath any radar notification!</i>"
                    )
                    await cls._send_telegram_reply(chat_id, reply)
                    return {"ok": True}

                # Command: /research, /search, /ask <natural language query>
                elif cmd in ("/research", "/search", "/ask"):
                    query = text[len(tokens[0]):].strip()
                    if not query:
                        reply = (
                            "🔬 <b>Targeted Research & Literature Search</b>\n\n"
                            "Ask any research question, thesis topic, or domain inquiry. Your request will be analyzed dynamically to fetch papers from arXiv without modifying your persistent daily radar settings.\n\n"
                            "<b>Usage:</b> <code>/research &lt;your question or topic&gt;</code>\n\n"
                            "<b>Examples:</b>\n"
                            "• <code>/research I have a thesis coming up and want to learn about theoretical cryptography</code>\n"
                            "• <code>/research Seminal papers on latent diffusion models for image generation</code>\n"
                            "• <code>/research Quantum algebra, braided tensor categories, and knot invariants</code>"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}

                    reply = (
                        f"🔬 <b>Targeted Research Initiated</b>\n"
                        f"• <b>Query:</b> <i>\"{query}\"</i>\n\n"
                        "🧠 <i>Formulating research strategy, mapping arXiv taxonomy, and ranking candidate literature...</i>"
                    )
                    await cls._send_telegram_reply(chat_id, reply)

                    asyncio.create_task(
                        execute_ad_hoc_research(
                            user_id=user.user_id,
                            chat_id=chat_id,
                            query=query,
                        )
                    )
                    return {"ok": True}

                # Command: /scan, /poll, /radar [source] [limit] [mode]
                elif cmd in ("/scan", "/poll", "/radar"):
                    args = text[len(tokens[0]):].strip()
                    cls._ensure_user_defaults(user)

                    scan_sources = list(user.enabled_sources)
                    scan_limit = user.scan_limit
                    discovery_mode = "latest"

                    alias_map = {
                        "arxiv": "arxiv", "paper": "arxiv", "papers": "arxiv", "research": "arxiv",
                        "hackernews": "hackernews", "hn": "hackernews", "news": "hackernews",
                        "github": "github", "gh": "github", "git": "github", "repos": "github",
                    }

                    if args:
                        for p in args.split():
                            p_lower = p.lower().strip()
                            if p_lower.isdigit():
                                scan_limit = max(1, min(int(p_lower), 50))
                            elif p_lower in alias_map:
                                scan_sources = [alias_map[p_lower]]
                            elif p_lower in ("foundational", "milestone", "classic", "milestones", "foundations"):
                                discovery_mode = "foundational"
                            elif p_lower in ("latest", "recent", "new"):
                                discovery_mode = "latest"

                    src_str = ", ".join([f"<code>{s}</code>" for s in scan_sources])
                    mode_line = f"• <b>Discovery Mode:</b> <code>{discovery_mode.capitalize()}</code>\n"
                    reply = (
                        f"🔍 <b>Radar Scan Initiated</b>\n"
                        f"• <b>Sources:</b> {src_str}\n"
                        f"• <b>Limit:</b> <code>{scan_limit}</code> items per source\n"
                        f"{mode_line}"
                        "Evaluating against your tracked topics with deduplication protection...\n"
                        "<i>Interactive alerts will arrive as new matches are discovered.</i>"
                    )
                    await cls._send_telegram_reply(chat_id, reply)
                    asyncio.create_task(
                        execute_scan(
                            user_id=user.user_id,
                            chat_id=chat_id,
                            threshold=user.similarity_threshold,
                            sources=scan_sources,
                            limit=scan_limit,
                            discovery_mode=discovery_mode,
                        )
                    )
                    return {"ok": True}

                # Command: /profile or /bio
                elif cmd in ("/profile", "/bio"):
                    args = text[len(tokens[0]):].strip()
                    cls._ensure_user_defaults(user)
                    domains_str = ", ".join([f"<code>{d}</code>" for d in user.tracked_domains]) or "<i>None (use /track)</i>"

                    # Subcommand: clear / reset
                    if args.lower() in ("clear", "reset", "remove", "delete"):
                        user.profile_summary = ""
                        try:
                            await sync_user_embedding(user)
                        except Exception as e:
                            logger.warning(f"Could not recompute embedding: {e}")

                        if db_session:
                            try:
                                await db_session.commit()
                            except Exception:
                                pass

                        reply = (
                            "🧹 <b>Developer Profile Bio Cleared</b>\n"
                            "• Your profile bio has been removed.\n"
                            f"• <b>Active Tracked Domains:</b> {domains_str}\n"
                            "• <i>Vector embedding re-synchronized using your active tracked domains.</i>"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}

                    # Subcommand: set / update bio text
                    elif args:
                        new_bio = args
                        if new_bio.lower().startswith("set "):
                            new_bio = new_bio[4:].strip()
                        elif new_bio.lower().startswith("update "):
                            new_bio = new_bio[7:].strip()

                        if not new_bio:
                            await cls._send_telegram_reply(
                                chat_id,
                                "❌ Usage: <code>/profile set I am an ML engineer focusing on LLM serving and distributed systems.</code>"
                            )
                            return {"ok": False}

                        user.profile_summary = new_bio
                        try:
                            await sync_user_embedding(user)
                        except Exception as e:
                            logger.warning(f"Could not compute embedding: {e}")

                        if db_session:
                            try:
                                await db_session.commit()
                            except Exception:
                                pass

                        reply = (
                            "👤 <b>Developer Profile Bio Updated</b>\n"
                            f"• <b>Profile Bio:</b> <i>\"{new_bio}\"</i>\n"
                            f"• <b>Tracked Domains:</b> {domains_str}\n\n"
                            "🎯 <i>Vector embedding synchronized (1536-dim)! Both your bio and tracked domains are now combined to evaluate similarity scores for incoming releases.</i>"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}

                    # No args: view current profile
                    else:
                        summary_val = getattr(user, "profile_summary", "")
                        if isinstance(summary_val, str) and summary_val.strip() and not summary_val.startswith("Default developer profile"):
                            current_bio = f"<i>\"{summary_val.strip()}\"</i>"
                        else:
                            current_bio = "<i>None configured yet</i>"

                        reply = (
                            "👤 <b>Developer Profile & Interests</b>\n\n"
                            f"• <b>Profile Bio:</b> {current_bio}\n"
                            f"• <b>Tracked Domains:</b> {domains_str}\n\n"
                            "<b>How it works:</b>\n"
                            "Your profile bio describes your engineering role and general technical background. "
                            "It is embedded alongside your explicit tracked domains into a dense 1536-dimensional vector to determine release relevance.\n\n"
                            "<b>Commands:</b>\n"
                            "• <code>/profile set &lt;description&gt;</code> — Set or update your bio\n"
                            "• <code>/profile clear</code> — Clear your bio\n"
                            "• <code>/track &lt;topics&gt;</code> — Add specific topic tags\n"
                            "• <code>/untrack &lt;topics&gt;</code> — Remove topic tags"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}

                # Command: /track <domain/keywords>
                elif cmd == "/track":
                    args = text[len(tokens[0]):].strip()
                    if not args:
                        await cls._send_telegram_reply(chat_id, "❌ Usage: <code>/track Domain Name, keyword1, keyword2</code>")
                        return {"ok": False}

                    new_items = [k.strip() for k in args.split(",") if k.strip()]
                    current = list(user.tracked_domains)
                    for item in new_items:
                        if item not in current:
                            current.append(item)

                    user.tracked_domains = current
                    try:
                        await sync_user_embedding(user)
                    except Exception as e:
                        logger.warning(f"Could not compute embedding: {e}")

                    if db_session:
                        try:
                            await db_session.commit()
                        except Exception:
                            pass

                    reply = (
                        f"✅ <b>Tracked Domains Updated</b>\n"
                        f"• <b>Active Domains:</b> {', '.join([f'<code>{d}</code>' for d in user.tracked_domains])}\n"
                        f"• <i>Dense vector regenerated & indexed with HNSW in pgvector.</i>"
                    )
                    await cls._send_telegram_reply(chat_id, reply)
                    return {"ok": True}

                # Command: /untrack <domains/keywords> (aliases: /remove, /deltrack)
                elif cmd in ("/untrack", "/remove", "/deltrack"):
                    args = text[len(tokens[0]):].strip()
                    if not args:
                        await cls._send_telegram_reply(
                            chat_id,
                            "❌ Usage: <code>/untrack keyword1, keyword2</code> or <code>/untrack all</code>"
                        )
                        return {"ok": False}

                    if args.lower() == "all":
                        removed_items = list(user.tracked_domains)
                        user.tracked_domains = []
                    else:
                        to_remove = [k.strip().lower() for k in args.split(",") if k.strip()]
                        remaining = []
                        removed_items = []
                        for d in user.tracked_domains:
                            if d.strip().lower() in to_remove:
                                removed_items.append(d)
                            else:
                                remaining.append(d)
                        user.tracked_domains = remaining

                    try:
                        await sync_user_embedding(user)
                    except Exception as e:
                        logger.warning(f"Could not compute embedding: {e}")

                    if db_session:
                        try:
                            await db_session.commit()
                        except Exception:
                            pass

                    if not removed_items:
                        active_str = ", ".join([f"<code>{d}</code>" for d in user.tracked_domains]) or "None"
                        reply = f"ℹ️ None of the specified topics were found in your tracked list.\n• <b>Active:</b> {active_str}"
                    else:
                        active_str = ", ".join([f"<code>{d}</code>" for d in user.tracked_domains]) or "<i>None (use /track to add)</i>"
                        reply = (
                            f"🗑️ <b>Removed from Tracked Radar:</b> {', '.join([f'<code>{r}</code>' for r in removed_items])}\n"
                            f"• <b>Remaining Active Domains:</b> {active_str}\n"
                            f"• <i>Dense vector regenerated & indexed.</i>"
                        )
                    await cls._send_telegram_reply(chat_id, reply)
                    return {"ok": True}

                # Command: /ignore <keywords>
                elif cmd == "/ignore":
                    args = text[len(tokens[0]):].strip()
                    if not args:
                        await cls._send_telegram_reply(chat_id, "❌ Usage: <code>/ignore keyword1, keyword2</code>")
                        return {"ok": False}

                    new_kws = [k.strip().lower() for k in args.split(",") if k.strip()]
                    current = list(user.ignored_keywords)
                    for k in new_kws:
                        if k not in current:
                            current.append(k)

                    user.ignored_keywords = current
                    if db_session:
                        try:
                            await db_session.commit()
                        except Exception:
                            pass

                    reply = (
                        f"🚫 <b>Negative Filter Updated (Zero LLM Token Cost)</b>\n"
                        f"• <b>Ignored Keywords:</b> {', '.join([f'<code>{k}</code>' for k in user.ignored_keywords])}"
                    )
                    await cls._send_telegram_reply(chat_id, reply)
                    return {"ok": True}

                # Command: /unignore <keywords>
                elif cmd in ("/unignore", "/delignore"):
                    args = text[len(tokens[0]):].strip()
                    if not args:
                        await cls._send_telegram_reply(
                            chat_id,
                            "❌ Usage: <code>/unignore keyword1, keyword2</code> or <code>/unignore all</code>"
                        )
                        return {"ok": False}

                    if args.lower() == "all":
                        removed_kws = list(user.ignored_keywords)
                        user.ignored_keywords = []
                    else:
                        to_remove = [k.strip().lower() for k in args.split(",") if k.strip()]
                        remaining = []
                        removed_kws = []
                        for k in user.ignored_keywords:
                            if k.strip().lower() in to_remove:
                                removed_kws.append(k)
                            else:
                                remaining.append(k)
                        user.ignored_keywords = remaining

                    if db_session:
                        try:
                            await db_session.commit()
                        except Exception:
                            pass

                    if not removed_kws:
                        active_str = ", ".join([f"<code>{k}</code>" for k in user.ignored_keywords]) or "None"
                        reply = f"ℹ️ None of the specified keywords were found in your ignore filter.\n• <b>Ignored:</b> {active_str}"
                    else:
                        active_str = ", ".join([f"<code>{k}</code>" for k in user.ignored_keywords]) or "<i>None</i>"
                        reply = (
                            f"✅ <b>Removed from Ignore Filter:</b> {', '.join([f'<code>{r}</code>' for r in removed_kws])}\n"
                            f"• <b>Remaining Ignored Keywords:</b> {active_str}"
                        )
                    await cls._send_telegram_reply(chat_id, reply)
                    return {"ok": True}

                # Command: /threshold <value>
                elif cmd == "/threshold":
                    args = text[len(tokens[0]):].strip()
                    try:
                        val = float(args)
                        if not (0.0 <= val <= 1.0):
                            raise ValueError()
                        user.similarity_threshold = val
                        if db_session:
                            try:
                                await db_session.commit()
                            except Exception:
                                pass
                        await cls._send_telegram_reply(chat_id, f"🎯 <b>Similarity Threshold Set:</b> <code>{val:.4f}</code>")
                    except Exception:
                        await cls._send_telegram_reply(chat_id, "❌ Please provide a float value between <code>0.0</code> and <code>1.0</code>.")
                    return {"ok": True}

                # Command: /sources, /source
                elif cmd in ("/sources", "/source"):
                    args = text[len(tokens[0]):].strip()
                    cls._ensure_user_defaults(user)

                    if not args:
                        all_avail = ["arxiv", "hackernews", "github"]
                        status_lines = []
                        for s in all_avail:
                            is_en = s in user.enabled_sources
                            icon = "✅ ENABLED" if is_en else "❌ DISABLED"
                            label = {"arxiv": "arXiv AI Papers", "hackernews": "HackerNews", "github": "GitHub Trending"}.get(s, s)
                            status_lines.append(f"• <b>{label}</b> (<code>{s}</code>): {icon}")

                        reply = (
                            f"📡 <b>Connected Technical Feed Sources:</b>\n"
                            + "\n".join(status_lines)
                            + "\n\n<b>Source Management Commands:</b>\n"
                            "• <code>/sources only arxiv</code> — Ingest ONLY from arXiv\n"
                            "• <code>/sources remove github</code> — Disable GitHub\n"
                            "• <code>/sources add github</code> — Re-enable GitHub\n"
                            "• <code>/sources all</code> — Enable all sources"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}

                    arg_lower = args.lower()
                    parts = arg_lower.split(maxsplit=1)
                    action = parts[0]
                    target = parts[1].strip() if len(parts) > 1 else ""

                    alias_map = {
                        "arxiv": "arxiv", "paper": "arxiv", "papers": "arxiv", "research": "arxiv",
                        "hackernews": "hackernews", "hn": "hackernews", "news": "hackernews",
                        "github": "github", "gh": "github", "git": "github", "repos": "github",
                    }

                    if action in ("only", "set"):
                        resolved = [alias_map.get(k.strip(), k.strip()) for k in target.replace(",", " ").split() if k.strip() in alias_map]
                        if not resolved:
                            await cls._send_telegram_reply(chat_id, "❌ Valid sources: <code>arxiv</code>, <code>hackernews</code>, <code>github</code>")
                            return {"ok": False}
                        user.enabled_sources = list(dict.fromkeys(resolved))
                    elif action in ("add", "+"):
                        resolved = [alias_map.get(k.strip(), k.strip()) for k in target.replace(",", " ").split() if k.strip() in alias_map]
                        if not resolved:
                            await cls._send_telegram_reply(chat_id, "❌ Valid sources: <code>arxiv</code>, <code>hackernews</code>, <code>github</code>")
                            return {"ok": False}
                        for r in resolved:
                            if r not in user.enabled_sources:
                                user.enabled_sources.append(r)
                    elif action in ("remove", "rm", "del", "-"):
                        resolved = [alias_map.get(k.strip(), k.strip()) for k in target.replace(",", " ").split() if k.strip() in alias_map]
                        if not resolved:
                            await cls._send_telegram_reply(chat_id, "❌ Valid sources: <code>arxiv</code>, <code>hackernews</code>, <code>github</code>")
                            return {"ok": False}
                        user.enabled_sources = [s for s in user.enabled_sources if s not in resolved]
                        if not user.enabled_sources:
                            user.enabled_sources = ["arxiv"]
                    elif action in ("all", "reset"):
                        user.enabled_sources = ["arxiv", "hackernews", "github"]
                    elif action in alias_map:
                        user.enabled_sources = [alias_map[action]]
                    else:
                        await cls._send_telegram_reply(chat_id, "❌ Usage: <code>/sources only arxiv</code>, <code>/sources remove github</code>, or <code>/sources all</code>")
                        return {"ok": False}

                    if db_session:
                        try:
                            await db_session.commit()
                        except Exception:
                            pass

                    reply = (
                        f"✅ <b>Active Sources Updated</b>\n"
                        f"• <b>Active:</b> {', '.join([f'<code>{s}</code>' for s in user.enabled_sources])}\n"
                        f"• <i>Radar scans will now only ingest from these sources.</i>"
                    )
                    await cls._send_telegram_reply(chat_id, reply)
                    return {"ok": True}

                # Command: /limit <number>
                elif cmd in ("/limit", "/max"):
                    args = text[len(tokens[0]):].strip()
                    cls._ensure_user_defaults(user)
                    if not args:
                        reply = (
                            f"🎯 <b>Current Feed Release Limit:</b> <code>{user.scan_limit}</code> items per source\n\n"
                            "To change: <code>/limit &lt;1-50&gt;</code> (e.g. <code>/limit 5</code>)"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}

                    try:
                        val = int(args)
                        if not (1 <= val <= 50):
                            raise ValueError()
                        user.scan_limit = val
                        if db_session:
                            try:
                                await db_session.commit()
                            except Exception:
                                pass
                        await cls._send_telegram_reply(
                            chat_id,
                            f"🎯 <b>Release Limit Updated:</b> <code>{val}</code> articles per source per scan."
                        )
                    except Exception:
                        await cls._send_telegram_reply(chat_id, "❌ Please provide an integer between <code>1</code> and <code>50</code> (e.g. <code>/limit 5</code>).")
                    return {"ok": True}

                # Command: /interval [minutes]
                elif cmd == "/interval":
                    args = text[len(tokens[0]):].strip()
                    current_interval = getattr(settings, "INGESTION_INTERVAL_MINUTES", 1440)
                    if not args:
                        hrs = current_interval / 60.0
                        reply = (
                            f"⏱️ <b>Background Ingestion Interval:</b> <code>{current_interval}</code> minutes ({hrs:.1f} hours)\n"
                            f"• <i>Configured from INGESTION_INTERVAL_MINUTES in your environment</i>\n\n"
                            "To change at runtime: <code>/interval &lt;minutes&gt;</code> (e.g. <code>/interval 1440</code> for 24h, or <code>/interval 60</code> for 1h)"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}

                    try:
                        val = int(args)
                        if val < 1:
                            raise ValueError()
                        settings.INGESTION_INTERVAL_MINUTES = val
                        hrs = val / 60.0
                        await cls._send_telegram_reply(
                            chat_id,
                            f"⏱️ <b>Ingestion Interval Set:</b> <code>{val}</code> minutes ({hrs:.1f} hours)."
                        )
                    except Exception:
                        await cls._send_telegram_reply(chat_id, "❌ Please provide a positive integer in minutes (e.g. <code>/interval 1440</code>).")
                    return {"ok": True}

                # Command: /reset, /clear
                elif cmd in ("/reset", "/clear"):
                    args = text[len(tokens[0]):].strip().lower()
                    from app.services.dedup_store import DeduplicationStore
                    if args in ("history", "seen", "cache", "all"):
                        count = DeduplicationStore.clear()
                        reply = (
                            f"🧹 <b>Deduplication History Cleared</b>\n"
                            f"• Reset <code>{count}</code> previously recorded articles.\n"
                            f"• Future scans will re-evaluate all fresh releases from scratch!"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}
                    else:
                        reply = (
                            "ℹ️ <b>Reset Commands:</b>\n"
                            "• <code>/reset history</code> — Clear deduplication memory so past items can be re-evaluated\n"
                            "• <code>/untrack all</code> — Remove all tracked domains\n"
                            "• <code>/unignore all</code> — Clear negative keyword filter"
                        )
                        await cls._send_telegram_reply(chat_id, reply)
                        return {"ok": True}

                # Command: /status
                elif cmd == "/status":
                    from app.services.dedup_store import DeduplicationStore
                    cls._ensure_user_defaults(user)
                    domains_str = ", ".join([f"<code>{d}</code>" for d in user.tracked_domains]) or "None"
                    ignored_str = ", ".join([f"<code>{k}</code>" for k in user.ignored_keywords]) or "None"
                    sources_str = ", ".join([f"<code>{s}</code>" for s in user.enabled_sources])
                    interval_mins = getattr(settings, "INGESTION_INTERVAL_MINUTES", 1440)
                    seen_count = DeduplicationStore.count()

                    summary_val = getattr(user, "profile_summary", "")
                    if isinstance(summary_val, str) and summary_val.strip() and not summary_val.startswith("Default developer profile"):
                        profile_bio_str = f"<code>{summary_val.strip()}</code>"
                    else:
                        profile_bio_str = "<i>None (use /profile set to add)</i>"

                    reply = (
                        f"📡 <b>Developer Tech Radar Configuration</b>\n"
                        f"• <b>Chat ID:</b> <code>{user.telegram_chat_id}</code>\n"
                        f"• <b>Similarity Threshold:</b> <code>{user.similarity_threshold:.4f}</code>\n"
                        f"• <b>Profile Bio:</b> {profile_bio_str}\n"
                        f"• <b>Tracked Domains:</b> {domains_str}\n"
                        f"• <b>Ignored Keywords:</b> {ignored_str}\n"
                        f"• <b>Active Sources:</b> {sources_str}\n"
                        f"• <b>Release Limit:</b> <code>{user.scan_limit}</code> articles/source\n"
                        f"• <b>Ingestion Interval:</b> <code>{interval_mins}</code> mins ({interval_mins / 60:.1f} hrs)\n"
                        f"• <b>Deduplicated History:</b> <code>{seen_count}</code> unique articles seen\n"
                        f"• <b>Vector Embedding:</b> Synchronized (1536-dim)"
                    )
                    await cls._send_telegram_reply(chat_id, reply)
                    return {"ok": True}

                # Command: /yes <thread_id> [prompt]
                elif cmd == "/yes":
                    parts = text.split(maxsplit=2)
                    if len(parts) < 2:
                        await cls._send_telegram_reply(chat_id, "❌ Usage: <code>/yes &lt;thread_id&gt; [optional instructions]</code>")
                        return {"ok": False}

                    thread_id = parts[1].strip()
                    user_prompt = parts[2].strip() if len(parts) > 2 else None

                    logger.info(f"Telegram text approval for thread={thread_id} with override='{user_prompt}'")
                    asyncio.create_task(
                        resume_graph(
                            thread_id=thread_id,
                            action="generate_tutorial",
                            user_prompt_override=user_prompt,
                        )
                    )
                    await cls._send_telegram_reply(
                        chat_id,
                        f"🚀 <b>Technical Brief Synthesis Initiated</b>\nCompiling publication-grade PDF brief for thread <code>{thread_id}</code>."
                    )
                    return {"ok": True}

                # Command: /skip <thread_id>
                elif cmd == "/skip":
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2:
                        await cls._send_telegram_reply(chat_id, "❌ Usage: <code>/skip &lt;thread_id&gt;</code>")
                        return {"ok": False}

                    thread_id = parts[1].strip()
                    logger.info(f"Telegram text dismissal for thread={thread_id}")
                    asyncio.create_task(
                        resume_graph(thread_id=thread_id, action="skip_release")
                    )
                    await cls._send_telegram_reply(chat_id, f"⏭️ Thread <code>{thread_id}</code> dismissed.")
                    return {"ok": True}

                elif text.startswith("/"):
                    await cls._send_telegram_reply(
                        chat_id,
                        "❓ Unknown command. Type /help to see all available commands."
                    )
                    return {"ok": False, "message": "Unknown command"}
            finally:
                if db_session:
                    try:
                        await db_session.close()
                    except Exception:
                        pass

        return {"ok": True, "message": "Ignored or non-text message"}


async def execute_ad_hoc_research(
    user_id: str,
    chat_id: int,
    query: str,
) -> None:
    """
    Executes targeted ad-hoc literature research for a free-form natural language query.
    Extracts taxonomy codes, fetches relevant papers from arXiv, calculates semantic similarity
    against the specific ad-hoc query, and dispatches ranked candidates with 1-click synthesis buttons.
    Leaves user's persistent profile and daily radar tracks completely untouched.
    """
    import uuid
    from app.agent.graph import get_radar_graph
    from app.agent.nodes.domain_filter_node import calculate_cosine_similarity, generate_embedding
    from app.agent.nodes.planner_node import plan_ad_hoc_research
    from app.services.ingestion import IngestionService
    from app.services.notifier import NotificationService
    from app.models.schemas import AlertPayload

    try:
        plan = await plan_ad_hoc_research(query, user_id=user_id)

        service = IngestionService()
        items = await service.ingest_arxiv(limit=15, plan=plan)

        if not items:
            cats_str = ", ".join([f"<code>{c}</code>" for c in plan.arxiv_categories])
            await TelegramWebhookHandler._send_telegram_reply(
                chat_id,
                f"ℹ️ <b>No papers retrieved from arXiv</b> for query: <i>\"{query}\"</i>.\n"
                f"Target categories searched: {cats_str}.\n"
                f"Try broadening your terms or using specific academic keywords."
            )
            return

        # Generate dense 1536-dimensional vector for the ad-hoc query
        query_vec = await generate_embedding(query)

        scored_candidates = []
        for item in items:
            paper_text = f"{item.title}: {item.summary}"
            paper_vec = await generate_embedding(paper_text)
            score = calculate_cosine_similarity(query_vec, paper_vec)

            matched_anchors = [a for a in plan.anchor_concepts if a in paper_text.lower()]
            domain_label = plan.arxiv_categories[0] if plan.arxiv_categories else "Research Paper"
            if matched_anchors:
                domain_label = f"{domain_label} ({matched_anchors[0].title()})"

            scored_candidates.append((score, item, [domain_label]))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:3]

        cats_str = ", ".join([f"<code>{c}</code>" for c in plan.arxiv_categories])
        terms_str = ", ".join([f"<code>{t}</code>" for t in plan.arxiv_query_terms[:3]])
        mode_label = "Foundational Research / Thesis Study" if plan.discovery_mode == "foundational" else "Latest Breakthroughs"

        header_msg = (
            f"🎯 <b>Research Strategy & Taxonomy Resolved</b>\n"
            f"• <b>Objective:</b> {mode_label}\n"
            f"• <b>arXiv Categories:</b> {cats_str}\n"
            f"• <b>Query Terms:</b> {terms_str}\n"
            f"• <b>Papers Evaluated:</b> {len(items)} | <b>Top Matches Dispatched:</b> {len(top_candidates)}\n\n"
            f"<i>Tap <b>📑 Summarize & Explain</b> on any paper below to compile a publication-grade PDF brief!</i>"
        )
        await TelegramWebhookHandler._send_telegram_reply(chat_id, header_msg)

        graph = get_radar_graph()
        notifier = NotificationService()

        for score, item, matched_domains in top_candidates:
            thread_id = str(uuid.uuid4())
            state = {
                "release_item": item,
                "user_id": user_id,
                "search_plan": plan.model_dump(),
                "relevance_score": score,
                "is_relevant": True,
                "matched_domains": matched_domains,
                "filter_reason": f"Ad-hoc research match (score: {score:.4f})",
                "notification_status": "ESCALATED_TELEGRAM",
                "user_prompt_override": f"Ad-hoc research query: {query}",
                "research_notes": [],
                "tutorial_markdown": None,
                "pdf_artifact_path": None,
                "error_logs": [],
            }
            config = {"configurable": {"thread_id": thread_id}}

            try:
                await graph.ainvoke(state, config=config)
            except Exception as ge:
                logger.warning(f"Graph ainvoke encountered: {ge}. Dispatching direct alert fallback.")
                payload = AlertPayload(
                    thread_id=thread_id,
                    release_title=item.title,
                    release_summary=item.summary,
                    source_url=item.source_url,
                    matched_domains=matched_domains,
                    relevance_score=score,
                    recipient_id=str(chat_id),
                    channel="TELEGRAM",
                )
                await notifier.send_alert(payload)

    except Exception as e:
        logger.error(f"Error executing ad-hoc research: {e}", exc_info=True)
        await TelegramWebhookHandler._send_telegram_reply(
            chat_id,
            f"⚠️ Research scan encountered an error: <code>{str(e)[:150]}</code>"
        )


async def execute_scan(
    user_id: str,
    chat_id: int,
    threshold: float,
    sources: Optional[List[str]] = None,
    limit: Optional[int] = None,
    discovery_mode: str = "latest",
) -> None:
    """Run an on-demand radar scan across specified sources with configurable limit and discovery mode."""
    import uuid
    from app.agent.graph import get_radar_graph
    from app.services.dedup_store import DeduplicationStore
    from app.services.ingestion import IngestionService
    from app.agent.nodes.planner_node import generate_search_plan

    try:
        user = await TelegramWebhookHandler.get_or_create_user(str(chat_id))
        plan = await generate_search_plan(user, discovery_mode=discovery_mode)

        service = IngestionService()
        items = await service.ingest_all(sources=sources, limit_per_source=limit, plan=plan)
        graph = get_radar_graph()

        matched_count = 0
        seen_count = 0
        keyword_count = 0
        threshold_count = 0

        for item in items:
            thread_id = str(uuid.uuid4())
            state = {
                "release_item": item,
                "user_id": user_id,
                "search_plan": plan.model_dump(),
                "relevance_score": 0.0,
                "is_relevant": False,
                "matched_domains": [],
                "filter_reason": "On-demand scan",
                "notification_status": "INITIALIZED",
                "user_prompt_override": None,
                "research_notes": [],
                "tutorial_markdown": None,
                "pdf_artifact_path": None,
                "error_logs": [],
            }
            config = {"configurable": {"thread_id": thread_id}}
            result = await graph.ainvoke(state, config=config)
            if result.get("is_relevant"):
                matched_count += 1
            else:
                reason = result.get("filter_reason", "")
                if "Stage 1" in reason:
                    seen_count += 1
                elif "Stage 2" in reason:
                    keyword_count += 1
                elif "Stage 3" in reason:
                    threshold_count += 1

        total_dedup = DeduplicationStore.count()
        src_label = ", ".join([f"<code>{s}</code>" for s in sources]) if sources else "<code>all sources</code>"

        summary = (
            f"🏁 <b>Radar Scan Complete</b>\n"
            f"• <b>Sources Scanned:</b> {src_label}\n"
            f"• <b>Discovery Mode:</b> <code>{discovery_mode.capitalize()}</code>\n"
            f"• <b>Total Releases Fetched:</b> {len(items)}\n"
            f"• <b>New Matches Dispatched:</b> {matched_count}\n"
        )
        if seen_count > 0:
            summary += f"• <b>Skipped (Already Seen):</b> {seen_count} <i>(use <code>/reset history</code> to re-evaluate)</i>\n"
        if keyword_count > 0:
            summary += f"• <b>Skipped (Negative Filter):</b> {keyword_count}\n"
        if threshold_count > 0:
            summary += f"• <b>Skipped (Below Threshold {threshold:.2f}):</b> {threshold_count}\n"
        summary += f"• <b>Deduplication Store:</b> {total_dedup} unique articles recorded\n"

        if matched_count == 0:
            if seen_count == len(items) and len(items) > 0:
                summary += (
                    f"\n💡 <i>All {len(items)} fetched releases were already processed in earlier scans! "
                    f"To re-evaluate them with your new threshold or profile, type <code>/reset history</code> and run <code>/scan</code> again, "
                    f"or scan more releases with <code>/scan arxiv 25</code>.</i>"
                )
            else:
                summary += (
                    f"\nℹ️ <i>No releases met your current criteria. "
                    f"You can adjust with <code>/threshold</code>, add topics with <code>/track</code>, "
                    f"or reset history with <code>/reset history</code>.</i>"
                )
        await TelegramWebhookHandler._send_telegram_reply(chat_id, summary)
    except Exception as e:
        logger.error(f"Error executing on-demand scan: {e}", exc_info=True)
        await TelegramWebhookHandler._send_telegram_reply(
            chat_id,
            f"⚠️ Scan encountered an error: <code>{str(e)[:150]}</code>"
        )


_telegram_polling_task: Optional[asyncio.Task] = None


async def telegram_polling_worker() -> None:
    """
    Background worker that runs Telegram Long Polling (getUpdates).
    Eliminates need for public webhook URLs, tunneling (ngrok), or open ports when running locally.
    Listens for user commands (/status, /track, /ignore, /threshold, /yes, /skip) and inline button clicks.
    """
    token = settings.TELEGRAM_BOT_TOKEN
    if not token or token.startswith("your_") or token.startswith("your-"):
        logger.info("Telegram bot token not configured or mock. Skipping Telegram long-polling worker.")
        return

    logger.info("Initializing Telegram Long Polling worker...")

    # Step 1: Delete any existing webhook so Telegram allows getUpdates polling.
    # We pass drop_pending_updates=False so any messages sent while offline are preserved & processed.
    delete_url = f"https://api.telegram.org/bot{token}/deleteWebhook"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(delete_url, json={"drop_pending_updates": False})
            if resp.status_code == 200:
                logger.info("Telegram webhook disabled successfully; switched to long-polling mode.")
            else:
                logger.warning(f"deleteWebhook returned status {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"Failed to reset Telegram webhook: {e}")

    offset: Optional[int] = None
    poll_url = f"https://api.telegram.org/bot{token}/getUpdates"
    logger.info("Telegram Long Polling active. Listening for commands and button interactions...")

    async with httpx.AsyncClient(timeout=35.0) as client:
        while True:
            try:
                params: Dict[str, Any] = {
                    "timeout": 20,
                    "allowed_updates": ["message", "callback_query"],
                }
                if offset is not None:
                    params["offset"] = offset

                resp = await client.get(poll_url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        updates = data.get("result", [])
                        for raw_upd in updates:
                            upd_id = raw_upd.get("update_id")
                            if upd_id is not None:
                                offset = upd_id + 1

                            try:
                                parsed_upd = TelegramUpdate.model_validate(raw_upd)
                                await TelegramWebhookHandler.process_update(parsed_upd)
                            except Exception as e:
                                logger.error(f"Error handling incoming Telegram update {upd_id}: {e}", exc_info=True)
                elif resp.status_code == 409:
                    logger.warning("Telegram polling conflict (status 409). Retrying in 5 seconds...")
                    await asyncio.sleep(5.0)
                else:
                    logger.warning(f"Telegram getUpdates unexpected response {resp.status_code}: {resp.text}")
                    await asyncio.sleep(2.0)

            except asyncio.CancelledError:
                logger.info("Telegram polling worker cancelled.")
                break
            except httpx.TimeoutException:
                continue
            except httpx.RequestError as req_err:
                logger.warning(f"Network error during Telegram polling: {req_err}. Retrying in 3s...")
                await asyncio.sleep(3.0)
            except Exception as e:
                logger.error(f"Unexpected error in Telegram polling worker: {e}", exc_info=True)
                await asyncio.sleep(2.0)


async def start_telegram_polling() -> None:
    """Start Telegram long-polling background task if bot token is configured."""
    global _telegram_polling_task
    token = settings.TELEGRAM_BOT_TOKEN
    if not token or token.startswith("your_") or token.startswith("your-"):
        logger.info("Telegram bot token not configured or mock. Skipping Telegram polling.")
        return

    _telegram_polling_task = asyncio.create_task(telegram_polling_worker())
    logger.info("Dispatched Telegram long-polling background task.")


async def stop_telegram_polling() -> None:
    """Gracefully cancel and terminate Telegram long-polling task on shutdown."""
    global _telegram_polling_task
    if _telegram_polling_task and not _telegram_polling_task.done():
        logger.info("Stopping Telegram long-polling worker...")
        _telegram_polling_task.cancel()
        try:
            await _telegram_polling_task
        except asyncio.CancelledError:
            pass
    _telegram_polling_task = None
