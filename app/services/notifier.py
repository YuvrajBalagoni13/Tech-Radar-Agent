"""
OOP Notification Subsystem implementing SOLID Principles and the Strategy Pattern.
Dispatches interactive embeds with action buttons to Discord and template alerts to Telegram via Bot API.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional
import httpx
from app.core.config import settings
from app.models.schemas import AlertPayload

logger = logging.getLogger("techradar.notifier")


class BaseNotifier(ABC):
    """
    Abstract Base Class (Dependency Inversion Principle).
    Defines the contract for dispatching real-time notifications and binary artifacts.
    """

    @abstractmethod
    async def send_notification(self, payload: AlertPayload) -> bool:
        """Dispatch structured alert to the communication channel."""
        pass

    @abstractmethod
    async def send_artifact(self, recipient: str, file_path: str, caption: str) -> bool:
        """Transmit compiled file artifact (e.g. PDF tutorial) to the target recipient."""
        pass


class DiscordNotifier(BaseNotifier):
    """
    Concrete Strategy for Discord Webhook and REST API interaction.
    Renders rich markdown embeds and interactive Action Row components (buttons).
    Supports Direct Messages (DMs) to individual users as well as server channel delivery.
    """

    _dm_channel_cache: Dict[str, str] = {}

    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.bot_token = bot_token if bot_token is not None else settings.DISCORD_BOT_TOKEN
        self.channel_id = channel_id if channel_id is not None else settings.DISCORD_CHANNEL_ID
        self.user_id = user_id if user_id is not None else settings.DISCORD_USER_ID
        self.timeout = timeout
        self.base_url = "https://discord.com/api/v10"

    def _determine_embed_color(self, score: float) -> int:
        """Generate hex color code reflecting algorithmic relevance score."""
        if score >= 0.90:
            return 0x2ECC71  # Vibrant Emerald
        elif score >= 0.85:
            return 0x3498DB  # Technical Cyan/Blue
        return 0xF39C12      # Alert Amber

    async def _get_or_create_dm_channel(self, user_snowflake: str) -> Optional[str]:
        """
        Open or retrieve an existing Direct Message (DM) channel with a Discord user.
        Uses Discord REST API: POST /users/@me/channels with {"recipient_id": user_snowflake}.
        Caches the resolved dm_channel_id in-memory for zero-overhead subsequent dispatches.
        """
        clean_user_id = str(user_snowflake).strip()
        if not clean_user_id or clean_user_id.startswith("default") or clean_user_id == "anonymous":
            return None

        if clean_user_id in self._dm_channel_cache:
            return self._dm_channel_cache[clean_user_id]

        if not self.bot_token or self.bot_token.startswith("your-") or self.bot_token.startswith("your_"):
            # Mock mode: synthesize a deterministic mock DM channel ID
            mock_dm_id = f"dm_{clean_user_id}"
            self._dm_channel_cache[clean_user_id] = mock_dm_id
            return mock_dm_id

        endpoint = f"{self.base_url}/users/@me/channels"
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
        }
        body = {"recipient_id": clean_user_id}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(endpoint, headers=headers, json=body)
                if resp.status_code == 200:
                    dm_data = resp.json()
                    dm_channel_id = dm_data.get("id")
                    if dm_channel_id:
                        self._dm_channel_cache[clean_user_id] = str(dm_channel_id)
                        logger.info(f"Resolved Discord DM channel {dm_channel_id} for user {clean_user_id}")
                        return str(dm_channel_id)
                else:
                    logger.warning(
                        f"Failed to create Discord DM channel for user {clean_user_id}: "
                        f"HTTP {resp.status_code} - {resp.text}"
                    )
        except Exception as e:
            logger.error(f"Exception creating Discord DM channel for user {clean_user_id}: {e}", exc_info=True)

        return None

    async def _resolve_target_channel(self, recipient: Optional[str] = None) -> Optional[str]:
        """
        Resolves the target Discord channel ID.
        Prioritizes:
        1. Recipient if it is a user ID (resolving DM channel)
        2. Configured DISCORD_USER_ID (resolving DM channel)
        3. Configured DISCORD_CHANNEL_ID (direct server channel delivery)
        """
        # Check target user ID candidates
        target_user = recipient if recipient and not recipient.startswith("default") else self.user_id
        if target_user:
            dm_channel = await self._get_or_create_dm_channel(target_user)
            if dm_channel:
                return dm_channel

        # Fall back to channel_id if configured or provided
        if recipient and recipient != "default_user" and not recipient.startswith("user_"):
            return recipient
        return self.channel_id or None

    async def send_notification(self, payload: AlertPayload) -> bool:
        """
        Dispatches a rich embed with interactive callback buttons to Discord.
        Delivers via Direct Message (DM) to user if configured, or falls back to server channel.
        """
        target_channel = await self._resolve_target_channel(payload.recipient_id)

        if (
            not self.bot_token
            or not target_channel
            or self.bot_token.startswith("your-")
            or str(target_channel).startswith("123456")
        ):
            logger.warning("Discord bot token or target recipient not configured. Mocking dispatch.")
            logger.info(f"[MOCK DISCORD] Dispatched alert for thread={payload.thread_id} to target={target_channel}: {payload.release_title}")
            return True

        embed = {
            "title": f"⚡ Tech Radar: {payload.release_title[:240]}",
            "url": payload.source_url,
            "description": payload.release_summary[:2000],
            "color": self._determine_embed_color(payload.relevance_score),
            "fields": [
                {
                    "name": "🎯 Semantic Score",
                    "value": f"`{payload.relevance_score:.4f}`",
                    "inline": True,
                },
                {
                    "name": "🏷️ Matched Domains",
                    "value": ", ".join([f"`{d}`" for d in payload.matched_domains]) if payload.matched_domains else "`General AI`",
                    "inline": True,
                },
                {
                    "name": "🧵 Workflow Thread",
                    "value": f"`{payload.thread_id}`",
                    "inline": True,
                },
            ],
            "footer": {
                "text": "Autonomous Tech Radar Agent • Click below to trigger deep research",
            },
        }

        # Discord ActionRow component with interactive buttons
        components = [
            {
                "type": 1,  # Action Row
                "components": [
                    {
                        "type": 2,  # Button
                        "style": 1,  # Primary (Blurple)
                        "label": "📑 Summarize & Explain",
                        "custom_id": f"action:generate:{payload.thread_id}",
                    },
                    {
                        "type": 2,  # Button
                        "style": 2,  # Secondary (Grey)
                        "label": "⏭️ Skip / Ignore",
                        "custom_id": f"action:skip:{payload.thread_id}",
                    },
                ],
            }
        ]

        endpoint = f"{self.base_url}/channels/{target_channel}/messages"
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
        }
        body = {
            "embeds": [embed],
            "components": components,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(endpoint, headers=headers, json=body)
                resp.raise_for_status()
                logger.info(f"Successfully posted interactive alert to Discord channel {target_channel}")
                return True
        except Exception as e:
            logger.error(f"Failed to deliver Discord notification: {e}", exc_info=True)
            return False

    async def send_artifact(self, recipient: str, file_path: str, caption: str) -> bool:
        """Upload generated PDF binary directly to the target Discord user DM or channel."""
        target_channel = await self._resolve_target_channel(recipient)
        if not self.bot_token or not target_channel:
            logger.warning("Discord credentials missing. Mocking PDF upload.")
            logger.info(f"[MOCK DISCORD] Uploaded artifact {file_path} to {target_channel} with caption '{caption}'")
            return True

        endpoint = f"{self.base_url}/channels/{target_channel}/messages"
        headers = {"Authorization": f"Bot {self.bot_token}"}

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            filename = file_path.split("/")[-1]
            files = {"file": (filename, file_bytes, "application/pdf")}
            data = {"content": f"📑 **Autonomous Tech Radar Synthesis**\n{caption}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(endpoint, headers=headers, data=data, files=files)
                resp.raise_for_status()
                logger.info(f"Uploaded PDF artifact {filename} to Discord channel {target_channel}")
                return True
        except Exception as e:
            logger.error(f"Failed to upload PDF artifact to Discord: {e}", exc_info=True)
            return False


class TelegramNotifier(BaseNotifier):
    """
    Concrete Strategy for Telegram Bot API communication.
    100% Free Forever, with support for rich HTML formatting, inline keyboards, and PDF uploads.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.bot_token = bot_token if bot_token is not None else settings.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id if chat_id is not None else settings.TELEGRAM_CHAT_ID
        self.timeout = timeout
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""

    async def send_notification(self, payload: AlertPayload) -> bool:
        """
        Dispatches rich HTML alert to Telegram chat with interactive inline action buttons.
        Buttons contain callback_data with LangGraph thread_id for 1-click resumption.
        """
        target_chat = None
        if payload.recipient_id:
            cand = str(payload.recipient_id).strip()
            if cand.startswith("telegram_"):
                cand = cand[len("telegram_"):]
            if cand.startswith("@") or cand.lstrip("-").isdigit():
                target_chat = cand
        if not target_chat:
            target_chat = self.chat_id

        if (
            not self.bot_token
            or not target_chat
            or self.bot_token.startswith("your_")
            or str(target_chat).startswith("your_")
        ):
            logger.warning("Telegram bot credentials not configured. Mocking Telegram alert.")
            logger.info(f"[MOCK TELEGRAM] Alert thread={payload.thread_id} to chat={target_chat}: {payload.release_title}")
            return True

        domains_str = ", ".join([f"<code>{d}</code>" for d in payload.matched_domains]) if payload.matched_domains else "<code>General AI</code>"
        message_text = (
            f"⚡ <b>TECH RADAR ALERT</b>\n\n"
            f"<b>Title:</b> {payload.release_title}\n"
            f"🎯 <b>Relevance:</b> <code>{payload.relevance_score:.4f}</code>\n"
            f"🏷️ <b>Domains:</b> {domains_str}\n"
            f"🔗 <a href=\"{payload.source_url}\">Read Source</a>\n\n"
            f"<b>Summary:</b>\n{payload.release_summary[:800]}...\n\n"
            f"🧵 <i>Thread: {payload.thread_id}</i>"
        )

        reply_markup = {
            "inline_keyboard": [
                [
                    {
                        "text": "📑 Summarize & Explain",
                        "callback_data": f"action:generate:{payload.thread_id}",
                    }
                ],
                [
                    {
                        "text": "⏭️ Skip / Ignore",
                        "callback_data": f"action:skip:{payload.thread_id}",
                    }
                ],
            ]
        }

        endpoint = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        body = {
            "chat_id": target_chat,
            "text": message_text,
            "parse_mode": "HTML",
            "reply_markup": reply_markup,
            "disable_web_page_preview": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(endpoint, json=body)
                resp.raise_for_status()
                logger.info(f"Successfully delivered alert to Telegram chat {target_chat}")
                return True
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram message: {e}", exc_info=True)
            return False

    async def send_artifact(self, recipient: str, file_path: str, caption: str) -> bool:
        """
        Uploads compiled PDF whitepaper directly to Telegram chat via sendDocument.
        Telegram allows free file uploads up to 50MB per document.
        """
        target_chat = None
        if recipient:
            cand = str(recipient).strip()
            if cand.startswith("telegram_"):
                cand = cand[len("telegram_"):]
            if cand.startswith("@") or cand.lstrip("-").isdigit():
                target_chat = cand
        if not target_chat:
            target_chat = self.chat_id

        if not self.bot_token or not target_chat or self.bot_token.startswith("your_"):
            logger.warning("Telegram credentials not configured. Mocking PDF upload.")
            logger.info(f"[MOCK TELEGRAM] Delivered artifact {file_path} to {target_chat}")
            return True

        endpoint = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"
        filename = file_path.split("/")[-1]

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            files = {"document": (filename, file_bytes, "application/pdf")}
            data = {
                "chat_id": target_chat,
                "caption": f"📑 <b>Autonomous Tech Radar Whitepaper</b>\n{caption}",
                "parse_mode": "HTML",
            }

            async with httpx.AsyncClient(timeout=35.0) as client:
                resp = await client.post(endpoint, data=data, files=files)
                resp.raise_for_status()
                logger.info(f"Delivered PDF artifact {filename} to Telegram chat {target_chat}")
                return True
        except Exception as e:
            logger.error(f"Failed to upload PDF artifact to Telegram: {e}", exc_info=True)
            return False


class NotificationService:
    """
    Context class for Strategy Pattern.
    Dynamically routes payloads to registered concrete notifiers without coupling graph nodes.
    Supports Discord and Telegram natively.
    """

    def __init__(self):
        self._strategies: Dict[str, BaseNotifier] = {
            "DISCORD": DiscordNotifier(),
            "TELEGRAM": TelegramNotifier(),
        }

    def register_strategy(self, channel_name: str, notifier: BaseNotifier) -> None:
        """Open-Closed Principle: Allows adding new channels (Slack, Email) at runtime."""
        self._strategies[channel_name.upper()] = notifier
        logger.info(f"Registered notification strategy for channel: {channel_name.upper()}")

    def get_notifier(self, channel_name: str) -> BaseNotifier:
        """Retrieve notifier instance with graceful fallback to Discord."""
        channel_key = channel_name.upper()
        if channel_key not in self._strategies:
            raise KeyError(f"Notification strategy '{channel_name}' is not registered.")
        return self._strategies[channel_key]

    async def send_alert(self, payload: AlertPayload) -> bool:
        """Route notification to appropriate channel strategy."""
        notifier = self.get_notifier(payload.channel)
        return await notifier.send_notification(payload)

    async def send_artifact(self, channel: str, recipient: str, file_path: str, caption: str) -> bool:
        """Route artifact transmission to appropriate channel strategy."""
        notifier = self.get_notifier(channel)
        return await notifier.send_artifact(recipient=recipient, file_path=file_path, caption=caption)
