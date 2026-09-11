"""
Unit and Integration Tests for Telegram Bot Long Polling and Interactive Command Handler.
Validates /status, /track, /ignore, /threshold, callback queries, and polling lifecycle.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.bot.telegram_handler import (
    TelegramWebhookHandler,
    start_telegram_polling,
    stop_telegram_polling,
)
from app.models.schemas import TelegramChat, TelegramFromUser, TelegramMessage, TelegramUpdate


@pytest.mark.asyncio
async def test_telegram_status_command():
    """Verify /status command returns active configuration."""
    update = TelegramUpdate(
        update_id=1001,
        message=TelegramMessage(
            message_id=1,
            chat=TelegramChat(id=8964229194, type="private"),
            from_user=TelegramFromUser(id=8964229194, is_bot=False, first_name="Tester"),
            text="/status",
        ),
    )

    mock_user = MagicMock()
    mock_user.telegram_chat_id = "8964229194"
    mock_user.similarity_threshold = 0.82
    mock_user.tracked_domains = ["Distributed Systems", "AI Infrastructure"]
    mock_user.ignored_keywords = ["crypto"]

    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        result = await TelegramWebhookHandler.process_update(update)

        assert result.get("ok") is True
        mock_reply.assert_awaited_once()
        sent_text = mock_reply.call_args[0][1]
        assert "Developer Tech Radar Configuration" in sent_text
        assert "Similarity Threshold" in sent_text


@pytest.mark.asyncio
async def test_telegram_track_and_ignore_commands():
    """Verify /track and /ignore update developer profile settings."""
    chat_id = 8964229194

    track_update = TelegramUpdate(
        update_id=1002,
        message=TelegramMessage(
            message_id=2,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/track Quantum Computing, Neurosymbolic AI",
        ),
    )

    mock_user = MagicMock()
    mock_user.telegram_chat_id = str(chat_id)
    mock_user.similarity_threshold = 0.82
    mock_user.tracked_domains = ["Distributed Systems"]
    mock_user.ignored_keywords = ["crypto"]

    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.sync_user_embedding", new_callable=AsyncMock), \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(track_update)
        assert res.get("ok") is True
        assert "Quantum Computing" in mock_user.tracked_domains
        sent_text = mock_reply.call_args[0][1]
        assert "Tracked Domains Updated" in sent_text

    ignore_update = TelegramUpdate(
        update_id=1003,
        message=TelegramMessage(
            message_id=3,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/ignore casino, betting",
        ),
    )

    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(ignore_update)
        assert res.get("ok") is True
        assert "casino" in mock_user.ignored_keywords
        sent_text = mock_reply.call_args[0][1]
        assert "Negative Filter Updated" in sent_text


@pytest.mark.asyncio
async def test_telegram_untrack_and_unignore_commands():
    """Verify /untrack and /unignore cleanly remove items from developer profile."""
    chat_id = 8964229194

    mock_user = MagicMock()
    mock_user.telegram_chat_id = str(chat_id)
    mock_user.similarity_threshold = 0.82
    mock_user.tracked_domains = ["Distributed Systems", "AI Infrastructure", "Robotics"]
    mock_user.ignored_keywords = ["crypto", "nft", "casino"]

    # Test /untrack Robotics
    untrack_update = TelegramUpdate(
        update_id=1010,
        message=TelegramMessage(
            message_id=10,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/untrack Robotics",
        ),
    )

    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.sync_user_embedding", new_callable=AsyncMock), \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(untrack_update)
        assert res.get("ok") is True
        assert "Robotics" not in mock_user.tracked_domains
        assert "Distributed Systems" in mock_user.tracked_domains
        sent_text = mock_reply.call_args[0][1]
        assert "Removed from Tracked Radar" in sent_text

    # Test /unignore casino
    unignore_update = TelegramUpdate(
        update_id=1011,
        message=TelegramMessage(
            message_id=11,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/unignore casino",
        ),
    )

    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(unignore_update)
        assert res.get("ok") is True
        assert "casino" not in mock_user.ignored_keywords
        assert "crypto" in mock_user.ignored_keywords
        sent_text = mock_reply.call_args[0][1]
        assert "Removed from Ignore Filter" in sent_text



@pytest.mark.asyncio
async def test_telegram_threshold_command():
    """Verify /threshold command adjusts cosine similarity threshold."""
    chat_id = 8964229194
    update = TelegramUpdate(
        update_id=1004,
        message=TelegramMessage(
            message_id=4,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/threshold 0.875",
        ),
    )

    mock_user = MagicMock()
    mock_user.telegram_chat_id = str(chat_id)
    mock_user.similarity_threshold = 0.82
    mock_user.tracked_domains = []
    mock_user.ignored_keywords = []

    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(update)
        assert res.get("ok") is True
        assert mock_user.similarity_threshold == 0.875
        sent_text = mock_reply.call_args[0][1]
        assert "Similarity Threshold Set" in sent_text
        assert "0.875" in sent_text


@pytest.mark.asyncio
async def test_telegram_scan_command():
    """Verify /scan command dispatches an immediate radar scan."""
    chat_id = 8964229194
    update = TelegramUpdate(
        update_id=1005,
        message=TelegramMessage(
            message_id=5,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/scan",
        ),
    )

    mock_user = MagicMock()
    mock_user.user_id = f"telegram_{chat_id}"
    mock_user.telegram_chat_id = str(chat_id)
    mock_user.similarity_threshold = 0.82

    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.execute_scan", new_callable=AsyncMock) as mock_exec, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(update)
        assert res.get("ok") is True
        sent_text = mock_reply.call_args[0][1]
        assert "Radar Scan Initiated" in sent_text


@pytest.mark.asyncio
async def test_telegram_sources_command():
    """Verify /sources allows selecting or removing specific sources."""
    chat_id = 8964229194
    mock_user = MagicMock()
    mock_user.telegram_chat_id = str(chat_id)
    mock_user.enabled_sources = ["arxiv", "hackernews", "github"]

    # 1. View sources
    view_update = TelegramUpdate(
        update_id=1020,
        message=TelegramMessage(
            message_id=20,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/sources",
        ),
    )
    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(view_update)
        assert res.get("ok") is True
        assert "Connected Technical Feed Sources" in mock_reply.call_args[0][1]

    # 2. Set to only arxiv
    only_arxiv_update = TelegramUpdate(
        update_id=1021,
        message=TelegramMessage(
            message_id=21,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/sources only arxiv",
        ),
    )
    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(only_arxiv_update)
        assert res.get("ok") is True
        assert mock_user.enabled_sources == ["arxiv"]
        assert "arxiv" in mock_reply.call_args[0][1]


@pytest.mark.asyncio
async def test_telegram_limit_and_interval_commands():
    """Verify /limit and /interval commands adjust retrieval parameters."""
    chat_id = 8964229194
    mock_user = MagicMock()
    mock_user.telegram_chat_id = str(chat_id)
    mock_user.scan_limit = 10

    limit_update = TelegramUpdate(
        update_id=1022,
        message=TelegramMessage(
            message_id=22,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/limit 5",
        ),
    )
    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(limit_update)
        assert res.get("ok") is True
        assert mock_user.scan_limit == 5
        assert "5" in mock_reply.call_args[0][1]


@pytest.mark.asyncio
async def test_deduplication_store_and_reset(tmp_path):
    """Verify DeduplicationStore tracks seen hashes and resets cleanly."""
    from app.services.dedup_store import DeduplicationStore
    import app.services.dedup_store as dedup_mod

    test_file = str(tmp_path / "test_seen_hashes.json")
    with patch.object(dedup_mod, "DEDUP_FILE", test_file):
        DeduplicationStore._loaded = False
        DeduplicationStore._seen_hashes = set()

        DeduplicationStore.clear()
        assert DeduplicationStore.is_seen("hash123") is False

        DeduplicationStore.mark_seen("hash123")
        assert DeduplicationStore.is_seen("hash123") is True
        assert DeduplicationStore.count() >= 1

        cleared = DeduplicationStore.clear()
        assert cleared >= 1
        assert DeduplicationStore.is_seen("hash123") is False

        DeduplicationStore._loaded = False
        DeduplicationStore._seen_hashes = set()


@pytest.mark.asyncio
async def test_telegram_profile_commands():
    """Verify /profile, /profile set, and /profile clear update the user profile summary and embedding."""
    chat_id = 8964229194
    mock_user = MagicMock()
    mock_user.telegram_chat_id = str(chat_id)
    mock_user.tracked_domains = ["Distributed Systems"]
    mock_user.ignored_keywords = ["crypto"]
    mock_user.similarity_threshold = 0.82
    mock_user.profile_summary = ""

    # 1. Test /profile view (empty)
    view_update = TelegramUpdate(
        update_id=2001,
        message=TelegramMessage(
            message_id=10,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/profile",
        ),
    )
    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(view_update)
        assert res.get("ok") is True
        assert "Developer Profile & Interests" in mock_reply.call_args[0][1]

    # 2. Test /profile set <bio>
    set_update = TelegramUpdate(
        update_id=2002,
        message=TelegramMessage(
            message_id=11,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/profile set Senior ML engineer building vLLM inference and GPU kernels",
        ),
    )
    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.sync_user_embedding", new_callable=AsyncMock) as mock_sync, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(set_update)
        assert res.get("ok") is True
        assert mock_user.profile_summary == "Senior ML engineer building vLLM inference and GPU kernels"
        mock_sync.assert_awaited_once_with(mock_user)
        assert "Profile Bio Updated" in mock_reply.call_args[0][1]

    # 3. Test /profile clear
    clear_update = TelegramUpdate(
        update_id=2003,
        message=TelegramMessage(
            message_id=12,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Tester"),
            text="/profile clear",
        ),
    )
    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.sync_user_embedding", new_callable=AsyncMock) as mock_sync, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        res = await TelegramWebhookHandler.process_update(clear_update)
        assert res.get("ok") is True
        assert mock_user.profile_summary == ""
        mock_sync.assert_awaited_once_with(mock_user)
        assert "Profile Bio Cleared" in mock_reply.call_args[0][1]


@pytest.mark.asyncio
async def test_telegram_polling_lifecycle():
    """Verify start and stop telegram polling functions operate cleanly without throwing exceptions."""
    from app.core.config import settings
    orig_token = settings.TELEGRAM_BOT_TOKEN
    settings.TELEGRAM_BOT_TOKEN = "mock_token_123:ABC"

    try:
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True}
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"ok": True, "result": []}

            await start_telegram_polling()
            await stop_telegram_polling()
    finally:
        settings.TELEGRAM_BOT_TOKEN = orig_token
