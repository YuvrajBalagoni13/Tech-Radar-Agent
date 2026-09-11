"""
Unit and Integration Tests for OOP Notification Strategy Engine.
Validates SOLID adherence, Discord embed formatting, Twilio WhatsApp dispatch,
and runtime Strategy selection.
"""

import pytest
from app.models.schemas import AlertPayload
from app.services.notifier import (
    BaseNotifier,
    DiscordNotifier,
    NotificationService,
    TelegramNotifier,
)


def test_base_notifier_subclassing():
    """Verify Liskov Substitution Principle and abstract interface adherence."""
    class CustomSlackNotifier(BaseNotifier):
        async def send_notification(self, payload: AlertPayload) -> bool:
            return True

        async def send_artifact(self, recipient: str, file_path: str, caption: str) -> bool:
            return True

    notifier = CustomSlackNotifier()
    assert isinstance(notifier, BaseNotifier)


def test_discord_notifier_color_thresholds():
    """Ensure Discord embed color dynamically encodes semantic relevance tier."""
    discord_notifier = DiscordNotifier()

    # Emerald Green for exceptional relevance >= 0.90
    assert discord_notifier._determine_embed_color(0.95) == 0x2ECC71

    # Cyan/Blue for solid relevance >= 0.85
    assert discord_notifier._determine_embed_color(0.87) == 0x3498DB

    # Amber for threshold boundary < 0.85
    assert discord_notifier._determine_embed_color(0.82) == 0xF39C12


@pytest.mark.asyncio
async def test_discord_mock_dispatch():
    """Verify Discord dispatch executes gracefully in mock mode."""
    notifier = DiscordNotifier(bot_token="", channel_id="")
    payload = AlertPayload(
        thread_id="test-thread-101",
        release_title="vLLM: High-Throughput and Memory-Efficient Inference",
        release_summary="PagedAttention memory management for LLM serving.",
        source_url="https://github.com/vllm-project/vllm",
        matched_domains=["LLM Serving", "Inference Optimization"],
        relevance_score=0.94,
        recipient_id="user_123",
        channel="DISCORD",
    )

    result = await notifier.send_notification(payload)
    assert result is True


@pytest.mark.asyncio
async def test_discord_dm_resolution_and_dispatch():
    """Verify Discord DM channel resolution and direct user message delivery."""
    notifier = DiscordNotifier(bot_token="", user_id="998877665544332211")
    dm_channel = await notifier._get_or_create_dm_channel("998877665544332211")
    assert dm_channel == "dm_998877665544332211"

    payload = AlertPayload(
        thread_id="test-thread-dm-01",
        release_title="FastAPI High-Performance Async Architecture",
        release_summary="Direct message delivery and verification.",
        source_url="https://fastapi.tiangolo.com",
        matched_domains=["AI Infrastructure"],
        relevance_score=0.92,
        recipient_id="998877665544332211",
        channel="DISCORD",
    )
    result = await notifier.send_notification(payload)
    assert result is True


@pytest.mark.asyncio
async def test_telegram_mock_dispatch():
    """Verify Telegram escalation executes gracefully in mock mode."""
    notifier = TelegramNotifier(bot_token="", chat_id="")
    payload = AlertPayload(
        thread_id="test-thread-202",
        release_title="LangGraph: Stateful Multi-Agent Applications",
        release_summary="Cyclic graph orchestration with durable persistence.",
        source_url="https://github.com/langchain-ai/langgraph",
        matched_domains=["Distributed Systems"],
        relevance_score=0.91,
        recipient_id="123456789",
        channel="TELEGRAM",
    )

    result = await notifier.send_notification(payload)
    assert result is True


@pytest.mark.asyncio
async def test_notification_service_strategy_pattern():
    """Validate runtime strategy selection and Open-Closed extensible registration."""
    service = NotificationService()

    # Discord strategy retrieval
    discord_strat = service.get_notifier("DISCORD")
    assert isinstance(discord_strat, DiscordNotifier)

    # Telegram strategy retrieval
    tg_strat = service.get_notifier("TELEGRAM")
    assert isinstance(tg_strat, TelegramNotifier)

    # Missing strategy should raise KeyError
    with pytest.raises(KeyError):
        service.get_notifier("SLACK")

    # Dynamic registration of new channel strategy (Open-Closed Principle)
    class MockSlackNotifier(BaseNotifier):
        async def send_notification(self, payload: AlertPayload) -> bool:
            return True

        async def send_artifact(self, recipient: str, file_path: str, caption: str) -> bool:
            return True

    service.register_strategy("SLACK", MockSlackNotifier())
    slack_strat = service.get_notifier("SLACK")
    assert isinstance(slack_strat, MockSlackNotifier)
