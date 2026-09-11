"""
Bot interaction package: Discord Client and Telegram Webhook Handler.
"""

from app.bot.discord_client import (
    DiscordInteractionHandler,
    TechRadarDiscordBot,
    start_discord_gateway,
    stop_discord_gateway,
    sync_user_embedding,
)
from app.bot.telegram_handler import (
    TelegramWebhookHandler,
    start_telegram_polling,
    stop_telegram_polling,
)

__all__ = [
    "DiscordInteractionHandler",
    "TechRadarDiscordBot",
    "start_discord_gateway",
    "stop_discord_gateway",
    "sync_user_embedding",
    "TelegramWebhookHandler",
    "start_telegram_polling",
    "stop_telegram_polling",
]
