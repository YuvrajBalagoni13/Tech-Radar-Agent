"""
Discord Client and Interaction Handler.
Implements Slash Commands (/radar track, /radar ignore, /radar threshold, /radar list),
Button Click Callbacks for LangGraph human-in-the-loop checkpoint resumption,
and Dynamic Embedding Synchronization in PostgreSQL with pgvector.
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from app.agent.graph import resume_graph
from app.agent.nodes.domain_filter_node import generate_embedding
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.entities import UserProfile

logger = logging.getLogger("techradar.discord_bot")


async def sync_user_embedding(user: UserProfile) -> List[float]:
    """
    Regenerate dense 1536-dimensional vector for developer profile and persist to PostgreSQL.
    Combines both personal profile bio/interests and explicit tracked domains into the vector.
    """
    domains_str = ", ".join(user.tracked_domains) if user.tracked_domains else "Software Architecture"
    summary = (getattr(user, "profile_summary", "") or "").strip()
    if summary and not summary.startswith("Default developer profile"):
        corpus = f"Developer Profile & Background: {summary}. Active Focus Areas: {domains_str}."
    else:
        corpus = f"Developer Profile: Active Focus Areas: {domains_str}."
    
    logger.info(f"Synchronizing dense vector embedding for user_id={user.user_id}...")
    new_embedding = await generate_embedding(corpus)
    user.profile_embedding = new_embedding
    return new_embedding


class DiscordInteractionHandler:
    """
    Processes Discord Gateway interactions and HTTP webhook callbacks.
    Executes slash command logic and resumes interrupted LangGraph checkpoint threads.
    """

    @staticmethod
    async def get_or_create_user(discord_id: str, session: Any) -> UserProfile:
        """Fetch user profile by Discord snowflake ID or provision a new developer entry."""
        stmt = select(UserProfile).where(UserProfile.discord_id == discord_id)
        res = await session.execute(stmt)
        user = res.scalars().first()

        if not user:
            user = UserProfile(
                user_id=f"discord_{discord_id}",
                discord_id=discord_id,
                tracked_domains=["Distributed Systems", "AI Infrastructure", "LLMs"],
                ignored_keywords=["crypto", "nft", "airdrop"],
                profile_summary="Default developer profile generated from Discord",
                similarity_threshold=settings.DEFAULT_SIMILARITY_THRESHOLD,
            )
            session.add(user)
            await session.flush()
            await sync_user_embedding(user)
            await session.commit()
            logger.info(f"Provisioned new UserProfile for Discord ID {discord_id}")

        return user

    @classmethod
    async def handle_slash_command(cls, name: str, options: Dict[str, Any], user_discord_id: str) -> Dict[str, Any]:
        """
        Execute /radar slash commands with dynamic embedding recalculation.
        """
        async with AsyncSessionLocal() as session:
            user = await cls.get_or_create_user(user_discord_id, session)

            if name == "track":
                domain = options.get("domain", "").strip()
                keywords_raw = options.get("keywords", "")
                new_keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

                current_domains = list(user.tracked_domains)
                if domain and domain not in current_domains:
                    current_domains.append(domain)
                for kw in new_keywords:
                    if kw not in current_domains:
                        current_domains.append(kw)

                user.tracked_domains = current_domains
                await sync_user_embedding(user)
                await session.commit()

                return {
                    "type": 4,  # ChannelMessageWithSource
                    "data": {
                        "content": (
                            f"✅ **Radar Tracking Updated**\n"
                            f"• **Added Domain:** `{domain}`\n"
                            f"• **Active Domains:** {', '.join([f'`{d}`' for d in user.tracked_domains])}\n"
                            f"• *Vector embedding regenerated and indexed with HNSW.*"
                        )
                    },
                }

            elif name == "ignore":
                keywords_raw = options.get("keywords", "")
                keywords = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
                current_ignored = list(user.ignored_keywords)

                added = []
                for kw in keywords:
                    if kw not in current_ignored:
                        current_ignored.append(kw)
                        added.append(kw)

                user.ignored_keywords = current_ignored
                await session.commit()

                return {
                    "type": 4,
                    "data": {
                        "content": (
                            f"🚫 **Negative Filter Updated (Zero LLM Token Cost)**\n"
                            f"• **Added Ignored Keywords:** {', '.join([f'`{k}`' for k in added])}\n"
                            f"• **Total Ignored Patterns:** {len(user.ignored_keywords)}"
                        )
                    },
                }

            elif name == "threshold":
                val = float(options.get("value", settings.DEFAULT_SIMILARITY_THRESHOLD))
                if not (0.0 <= val <= 1.0):
                    return {
                        "type": 4,
                        "data": {"content": "❌ **Error**: Threshold must be between `0.0` and `1.0`."},
                    }

                user.similarity_threshold = val
                await session.commit()
                return {
                    "type": 4,
                    "data": {
                        "content": f"🎯 **Cosine Similarity Threshold Set:** `{val:.4f}`"
                    },
                }

            elif name == "list":
                domains_str = ", ".join([f"`{d}`" for d in user.tracked_domains]) or "None"
                ignored_str = ", ".join([f"`{k}`" for k in user.ignored_keywords]) or "None"
                return {
                    "type": 4,
                    "data": {
                        "content": (
                            f"📡 **Active Developer Tech Radar Configuration**\n"
                            f"• **User ID:** `{user.user_id}`\n"
                            f"• **Tracked Domains:** {domains_str}\n"
                            f"• **Ignored Keywords:** {ignored_str}\n"
                            f"• **Similarity Threshold:** `{user.similarity_threshold:.4f}`\n"
                            f"• **Vector Embedding Status:** `Synchronized (1536 dims)`"
                        )
                    },
                }

            elif name in ("research", "search"):
                query = str(options.get("query", "")).strip()
                if not query:
                    return {
                        "type": 4,
                        "data": {
                            "content": (
                                "🔬 **Ad-Hoc Targeted Research**\n"
                                "Please specify a query (e.g. `/radar research query: I have a thesis coming and want to learn about theoretical cryptography`)."
                            )
                        },
                    }

                import asyncio
                import uuid
                from app.agent.nodes.planner_node import plan_ad_hoc_research
                from app.services.ingestion import IngestionService
                from app.agent.nodes.domain_filter_node import calculate_cosine_similarity, generate_embedding
                from app.services.notifier import NotificationService
                from app.models.schemas import AlertPayload

                async def _run_discord_research():
                    try:
                        plan = await plan_ad_hoc_research(query, user_id=user.user_id)
                        service = IngestionService()
                        items = await service.ingest_arxiv(limit=10, plan=plan)
                        if not items:
                            return

                        query_vec = await generate_embedding(query)
                        scored = []
                        for it in items:
                            txt = f"{it.title}: {it.summary}"
                            pv = await generate_embedding(txt)
                            sc = calculate_cosine_similarity(query_vec, pv)
                            scored.append((sc, it))

                        scored.sort(key=lambda x: x[0], reverse=True)
                        notifier = NotificationService()
                        for sc, it in scored[:3]:
                            tid = str(uuid.uuid4())
                            payload = AlertPayload(
                                thread_id=tid,
                                release_title=it.title,
                                release_summary=it.summary,
                                source_url=it.source_url,
                                matched_domains=plan.arxiv_categories[:2],
                                relevance_score=sc,
                                recipient_id=user_discord_id,
                                channel="DISCORD",
                            )
                            await notifier.send_alert(payload)
                    except Exception as exc:
                        logger.error(f"Error in Discord ad-hoc research: {exc}")

                asyncio.create_task(_run_discord_research())

                return {
                    "type": 4,
                    "data": {
                        "content": (
                            f"🔬 **Targeted Research Scan Initiated**\n"
                            f"• **Query:** *\"{query}\"*\n"
                            "🧠 Formulating search plan and ranking literature from arXiv..."
                        )
                    },
                }

            return {
                "type": 4,
                "data": {"content": f"Unknown slash command: `{name}`"},
            }

    @classmethod
    async def handle_button_click(cls, custom_id: str, user_discord_id: str) -> Dict[str, Any]:
        """
        Handle button interaction callbacks: action:generate:<thread_id> or action:skip:<thread_id>.
        Resumes the LangGraph state machine from its checkpointed boundary.
        """
        parts = custom_id.split(":")
        if len(parts) < 3:
            return {
                "type": 4,
                "data": {"content": "❌ Invalid interaction identifier."},
            }

        action_type = parts[1]  # 'generate' or 'skip'
        thread_id = parts[2]

        if action_type == "generate":
            logger.info(f"User approved tutorial generation for thread '{thread_id}' via Discord button.")
            # Trigger asynchronous workflow resumption in the background
            import asyncio
            asyncio.create_task(resume_graph(thread_id, action="generate_tutorial"))

            return {
                "type": 4,
                "data": {
                    "content": (
                        f"🚀 **Action Authorized for Thread `{thread_id}`**\n"
                        f"Conducting domain-scoped deep research and compiling publication-grade PDF whitepaper. "
                        f"You will receive the compiled PDF momentarily!"
                    )
                },
            }

        elif action_type == "skip":
            logger.info(f"User dismissed item for thread '{thread_id}' via Discord button.")
            import asyncio
            asyncio.create_task(resume_graph(thread_id, action="skip_release"))

            return {
                "type": 4,
                "data": {
                    "content": f"⏭️ **Item Dismissed.** Workflow thread `{thread_id}` has been safely closed."
                },
            }

        return {
            "type": 4,
            "data": {"content": f"Unrecognized action type: `{action_type}`"},
        }


import asyncio
import discord
from discord.ext import commands


class TechRadarDiscordBot(commands.Bot):
    """
    Discord Gateway Bot supporting Direct Messages (DMs), slash commands,
    and button interactions directly within private user conversations.
    """

    def __init__(self, intents: Optional[discord.Intents] = None):
        if intents is None:
            intents = discord.Intents.default()
            intents.message_content = True
        super().__init__(command_prefix="!", intents=intents, help_command=None)

    async def on_ready(self):
        logger.info(f"Discord Gateway Bot successfully authenticated as {self.user} (ID: {self.user.id})")

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        # Check if message is in a Direct Message (DM)
        if isinstance(message.channel, discord.DMChannel) or message.guild is None:
            await self._handle_dm_message(message)
            return

        await self.process_commands(message)

    async def _handle_dm_message(self, message: discord.Message):
        text = message.content.strip()
        user_id = str(message.author.id)

        async with AsyncSessionLocal() as session:
            user = await DiscordInteractionHandler.get_or_create_user(user_id, session)

            if text.startswith("!track") or text.startswith("/track"):
                args = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
                if not args:
                    await message.channel.send("❌ Usage: `!track Distributed Systems, AI Infrastructure`")
                    return

                new_domains = [d.strip() for d in args.split(",") if d.strip()]
                curr = list(user.tracked_domains)
                for d in new_domains:
                    if d not in curr:
                        curr.append(d)
                user.tracked_domains = curr
                await sync_user_embedding(user)
                await session.commit()

                await message.channel.send(
                    f"✅ **Radar Tracking Updated in DM**\n"
                    f"• **Active Domains:** {', '.join([f'`{d}`' for d in user.tracked_domains])}\n"
                    f"• *Vector embedding regenerated and indexed with HNSW.*"
                )

            elif text.startswith("!untrack") or text.startswith("/untrack") or text.startswith("!remove"):
                args = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
                if not args:
                    await message.channel.send("❌ Usage: `!untrack keyword1, keyword2` or `!untrack all`")
                    return

                if args.lower() == "all":
                    removed = list(user.tracked_domains)
                    user.tracked_domains = []
                else:
                    to_remove = [k.strip().lower() for k in args.split(",") if k.strip()]
                    remaining = []
                    removed = []
                    for d in user.tracked_domains:
                        if d.strip().lower() in to_remove:
                            removed.append(d)
                        else:
                            remaining.append(d)
                    user.tracked_domains = remaining

                await sync_user_embedding(user)
                await session.commit()

                active_str = ", ".join([f"`{d}`" for d in user.tracked_domains]) or "*None*"
                await message.channel.send(
                    f"🗑️ **Removed from Tracked Footprint:** {', '.join([f'`{r}`' for r in removed])}\n"
                    f"• **Active Domains:** {active_str}"
                )

            elif text.startswith("!ignore") or text.startswith("/ignore"):
                args = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
                if not args:
                    await message.channel.send("❌ Usage: `!ignore crypto, nft, airdrop`")
                    return

                kws = [k.strip().lower() for k in args.split(",") if k.strip()]
                curr = list(user.ignored_keywords)
                for k in kws:
                    if k not in curr:
                        curr.append(k)
                user.ignored_keywords = curr
                await session.commit()

                await message.channel.send(
                    f"🚫 **Negative Filter Updated (Zero LLM Token Cost)**\n"
                    f"• **Ignored Patterns:** {', '.join([f'`{k}`' for k in user.ignored_keywords])}"
                )

            elif text.startswith("!unignore") or text.startswith("/unignore"):
                args = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
                if not args:
                    await message.channel.send("❌ Usage: `!unignore keyword1, keyword2` or `!unignore all`")
                    return

                if args.lower() == "all":
                    removed = list(user.ignored_keywords)
                    user.ignored_keywords = []
                else:
                    to_remove = [k.strip().lower() for k in args.split(",") if k.strip()]
                    remaining = []
                    removed = []
                    for k in user.ignored_keywords:
                        if k.strip().lower() in to_remove:
                            removed.append(k)
                        else:
                            remaining.append(k)
                    user.ignored_keywords = remaining

                await session.commit()
                active_str = ", ".join([f"`{k}`" for k in user.ignored_keywords]) or "*None*"
                await message.channel.send(
                    f"✅ **Removed from Negative Filter:** {', '.join([f'`{r}`' for r in removed])}\n"
                    f"• **Ignored Patterns:** {active_str}"
                )

            elif text.startswith("!threshold") or text.startswith("/threshold"):
                args = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
                try:
                    val = float(args)
                    if not (0.0 <= val <= 1.0):
                        raise ValueError()
                    user.similarity_threshold = val
                    await session.commit()
                    await message.channel.send(f"🎯 **Cosine Similarity Threshold Set:** `{val:.4f}`")
                except Exception:
                    await message.channel.send("❌ Please provide a float between `0.0` and `1.0` (e.g. `!threshold 0.85`).")

            elif text.startswith("!profile") or text.startswith("/profile") or text.startswith("!bio"):
                parts = text.split(maxsplit=1)
                args = parts[1].strip() if len(parts) > 1 else ""
                domains_str = ", ".join([f"`{d}`" for d in user.tracked_domains]) or "*None (use !track)*"

                if args.lower() in ("clear", "reset"):
                    user.profile_summary = ""
                    await sync_user_embedding(user)
                    await session.commit()
                    await message.channel.send(
                        "🧹 **Developer Profile Bio Cleared**\n"
                        "• Profile bio has been reset.\n"
                        f"• **Active Tracked Domains:** {domains_str}\n"
                        "• *Vector embedding re-synchronized using active tracked domains.*"
                    )
                elif args:
                    new_bio = args
                    if new_bio.lower().startswith("set "):
                        new_bio = new_bio[4:].strip()
                    user.profile_summary = new_bio
                    await sync_user_embedding(user)
                    await session.commit()
                    await message.channel.send(
                        "👤 **Developer Profile Bio Updated**\n"
                        f"• **Profile Bio:** *\"{new_bio}\"*\n"
                        f"• **Tracked Domains:** {domains_str}\n"
                        "🎯 *Dense 1536-dimensional vector embedding synchronized! Both your bio and tracked domains are used for similarity scoring.*"
                    )
                else:
                    bio_val = getattr(user, "profile_summary", "") or ""
                    bio_disp = f"*\"{bio_val}\"*" if bio_val and not bio_val.startswith("Default developer profile") else "*None configured*"
                    await message.channel.send(
                        "👤 **Developer Profile & Technical Interests**\n"
                        f"• **Profile Bio:** {bio_disp}\n"
                        f"• **Tracked Domains:** {domains_str}\n\n"
                        "**Commands:**\n"
                        "• `!profile set <bio>` - Set your background/role description\n"
                        "• `!profile clear` - Reset your profile bio"
                    )

            elif text in ("!status", "/status", "!list", "/list"):
                domains_str = ", ".join([f"`{d}`" for d in user.tracked_domains]) or "None"
                ignored_str = ", ".join([f"`{k}`" for k in user.ignored_keywords]) or "None"
                bio_val = getattr(user, "profile_summary", "") or ""
                bio_disp = f"`{bio_val}`" if bio_val and not bio_val.startswith("Default developer profile") else "*None (use !profile set to add)*"
                await message.channel.send(
                    f"📡 **Active Developer Tech Radar Configuration**\n"
                    f"• **Discord User ID:** `{user.discord_id}`\n"
                    f"• **Profile Bio:** {bio_disp}\n"
                    f"• **Tracked Domains:** {domains_str}\n"
                    f"• **Ignored Keywords:** {ignored_str}\n"
                    f"• **Similarity Threshold:** `{user.similarity_threshold:.4f}`\n"
                    f"• **Vector Embedding:** `Synchronized (1536 dims)`"
                )

            elif text.startswith("!yes") or text.startswith("/yes"):
                parts = text.split(maxsplit=2)
                if len(parts) < 2:
                    await message.channel.send("❌ Usage: `!yes <thread_id> [optional prompt]`")
                    return
                thread_id = parts[1]
                override = parts[2] if len(parts) > 2 else None
                await message.channel.send(f"🚀 Authorizing tutorial generation for thread `{thread_id}`...")
                asyncio.create_task(
                    resume_graph(thread_id, action="generate_tutorial", user_prompt_override=override)
                )

            elif text.startswith("!skip") or text.startswith("/skip"):
                parts = text.split(maxsplit=1)
                if len(parts) < 2:
                    await message.channel.send("❌ Usage: `!skip <thread_id>`")
                    return
                thread_id = parts[1]
                await message.channel.send(f"⏭️ Thread `{thread_id}` dismissed.")
                asyncio.create_task(
                    resume_graph(thread_id, action="skip_release")
                )

            elif text in ("!help", "/help", "!start", "/start"):
                await message.channel.send(
                    "🤖 **Autonomous Tech Radar Bot (Direct Message Mode)**\n\n"
                    "Available DM Commands:\n"
                    "• `!profile` - View developer profile bio & interests\n"
                    "• `!profile set <bio>` - Describe background/role for AI similarity matching\n"
                    "• `!profile clear` - Remove profile bio\n"
                    "• `!track <domains>` - Add domains to your radar footprint\n"
                    "• `!ignore <keywords>` - Add negative keywords (zero LLM token cost)\n"
                    "• `!threshold <0.0-1.0>` - Adjust cosine similarity threshold\n"
                    "• `!status` - View your active profile and vector sync state\n"
                    "• `!yes <thread_id> [notes]` - Authorize tutorial generation\n"
                    "• `!skip <thread_id>` - Dismiss release item\n\n"
                    "*(Interactive buttons on radar alert embeds can also be clicked directly in this DM!)*"
                )


_discord_bot_instance: Optional[TechRadarDiscordBot] = None
_discord_bot_task: Optional[asyncio.Task] = None


async def start_discord_gateway() -> None:
    """Start Discord Gateway connection in the background if token configured."""
    global _discord_bot_instance, _discord_bot_task
    token = settings.DISCORD_BOT_TOKEN
    if not token or token.startswith("your-") or token.startswith("your_"):
        logger.info("Discord bot token not configured or mock. Skipping Discord gateway runner.")
        return

    async def _runner():
        global _discord_bot_instance
        try:
            intents = discord.Intents.default()
            intents.message_content = True
            _discord_bot_instance = TechRadarDiscordBot(intents=intents)
            logger.info("Connecting Discord Gateway Bot for Direct Messages...")
            await _discord_bot_instance.start(token)
        except discord.errors.PrivilegedIntentsRequired:
            logger.warning(
                "Privileged message_content intent not enabled in Discord Developer Portal. "
                "Retrying with default gateway intents..."
            )
            try:
                _discord_bot_instance = TechRadarDiscordBot(intents=discord.Intents.default())
                await _discord_bot_instance.start(token)
            except Exception as e:
                logger.warning(f"Discord gateway connection failed with default intents: {e}")
        except Exception as e:
            logger.warning(f"Discord gateway connection failed: {e}")

    _discord_bot_task = asyncio.create_task(_runner())


async def stop_discord_gateway() -> None:
    """Gracefully disconnect Discord Gateway bot on application shutdown."""
    global _discord_bot_instance, _discord_bot_task
    if _discord_bot_instance and not _discord_bot_instance.is_closed():
        try:
            logger.info("Disconnecting Discord Gateway bot...")
            await _discord_bot_instance.close()
        except Exception as e:
            logger.warning(f"Error disconnecting Discord gateway: {e}")

    if _discord_bot_task and not _discord_bot_task.done():
        _discord_bot_task.cancel()
        try:
            await _discord_bot_task
        except asyncio.CancelledError:
            pass
