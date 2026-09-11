"""
Unit tests for Ad-Hoc Research Planner and /research, /search commands.
Validates natural language intent detection, arXiv taxonomy mapping,
and non-destructive ad-hoc querying decoupled from persistent profiles.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.agent.nodes.planner_node import plan_ad_hoc_research
from app.bot.discord_client import DiscordInteractionHandler
from app.bot.telegram_handler import TelegramWebhookHandler, execute_ad_hoc_research
from app.models.entities import UserProfile
from app.models.schemas import (
    TechReleaseItem,
    TelegramChat,
    TelegramFromUser,
    TelegramMessage,
    TelegramUpdate,
)


@pytest.mark.asyncio
async def test_plan_ad_hoc_research_thesis_cryptography():
    """Verify natural language thesis intent resolves to foundational mode and cs.CR."""
    query = "i have thesis comming & want to learn about theoretical cryptography so want research accordingly"
    with patch("app.core.config.settings.GROQ_API_KEY", ""):
        plan = await plan_ad_hoc_research(query)

    assert plan.discovery_mode == "foundational"
    assert plan.arxiv_sort_by == "relevance"
    assert "cs.CR" in plan.arxiv_categories
    assert any("cryptography" in term for term in plan.arxiv_query_terms)
    assert any("cipher" in a or "cryptography" in a or "encryption" in a for a in plan.anchor_concepts)


@pytest.mark.asyncio
async def test_plan_ad_hoc_research_latest_image_generation():
    """Verify 'latest' breakthrough query resolves to latest mode and cs.CV."""
    query = "give me the latest image generation research papers"
    with patch("app.core.config.settings.GROQ_API_KEY", ""):
        plan = await plan_ad_hoc_research(query)

    assert plan.discovery_mode == "latest"
    assert plan.arxiv_sort_by == "submittedDate"
    assert "cs.CV" in plan.arxiv_categories
    assert any("diffusion" in term or "image generation" in term for term in plan.arxiv_query_terms)


@pytest.mark.asyncio
async def test_plan_ad_hoc_research_quantum_algebra():
    """Verify quantum algebra intent resolves to math.QA / quant-ph."""
    query = "foundations of quantum algebra and braided categories"
    with patch("app.core.config.settings.GROQ_API_KEY", ""):
        plan = await plan_ad_hoc_research(query)

    assert plan.discovery_mode == "foundational"
    assert any(c in plan.arxiv_categories for c in ("math.QA", "quant-ph"))
    assert any("algebra" in term or "quantum" in term for term in plan.arxiv_query_terms)


@pytest.mark.asyncio
async def test_telegram_research_command_preserves_persistent_profile():
    """Verify /research command does NOT alter the user's persistent tracked domains or bio."""
    chat_id = 99887766
    update = TelegramUpdate(
        update_id=2001,
        message=TelegramMessage(
            message_id=50,
            chat=TelegramChat(id=chat_id, type="private"),
            from_user=TelegramFromUser(id=chat_id, is_bot=False, first_name="Researcher"),
            text="/research i have thesis comming & want to learn about theoretical cryptography",
        ),
    )

    mock_user = UserProfile(
        user_id=f"telegram_{chat_id}",
        telegram_chat_id=str(chat_id),
        tracked_domains=["Machine Learning", "LLM Serving"],
        profile_summary="Senior ML Engineer",
        similarity_threshold=0.85,
    )

    with patch.object(TelegramWebhookHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply, \
         patch("app.bot.telegram_handler.execute_ad_hoc_research", new_callable=AsyncMock) as mock_exec, \
         patch("app.bot.telegram_handler.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user
        mock_reply.return_value = True

        result = await TelegramWebhookHandler.process_update(update)

        assert result.get("ok") is True
        mock_reply.assert_awaited_once()
        sent_text = mock_reply.call_args[0][1]
        assert "Targeted Research Initiated" in sent_text
        assert "theoretical cryptography" in sent_text

        # Verify persistent profile was NOT modified
        assert mock_user.tracked_domains == ["Machine Learning", "LLM Serving"]
        assert mock_user.profile_summary == "Senior ML Engineer"


@pytest.mark.asyncio
async def test_execute_ad_hoc_research_flow():
    """Verify execute_ad_hoc_research ingests, ranks against query, and dispatches alerts."""
    chat_id = 99887766
    user_id = f"telegram_{chat_id}"
    query = "theoretical cryptography and zero knowledge"

    mock_papers = [
        TechReleaseItem(
            title="Foundations of Theoretical Cryptography and ZK Proofs",
            summary="A comprehensive study of zero-knowledge proofs and post-quantum encryption.",
            source_url="https://arxiv.org/abs/2103.11244",
            source_type="arxiv",
        ),
        TechReleaseItem(
            title="Accelerating Image Generation With Latent Models",
            summary="Denoising diffusion models applied to image synthesis.",
            source_url="https://arxiv.org/abs/2112.10752",
            source_type="arxiv",
        ),
    ]

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"is_relevant": True})

    with patch("app.services.ingestion.IngestionService.ingest_arxiv", new_callable=AsyncMock) as mock_ingest, \
         patch("app.agent.graph.get_radar_graph", return_value=mock_graph), \
         patch.object(TelegramWebhookHandler, "_send_telegram_reply", new_callable=AsyncMock) as mock_reply:
        mock_ingest.return_value = mock_papers
        mock_reply.return_value = True

        await execute_ad_hoc_research(user_id=user_id, chat_id=chat_id, query=query)

        # Confirm strategy summary was sent
        assert mock_reply.call_count >= 1
        header_text = mock_reply.call_args_list[0][0][1]
        assert "Research Strategy & Taxonomy Resolved" in header_text

        # Confirm graph was invoked for top candidates
        assert mock_graph.ainvoke.call_count >= 1
        invoked_state = mock_graph.ainvoke.call_args[0][0]
        assert invoked_state["is_relevant"] is True
        assert invoked_state["relevance_score"] > 0.0


@pytest.mark.asyncio
async def test_discord_research_slash_command():
    """Verify Discord /radar research command executes cleanly."""
    mock_user = UserProfile(
        user_id="discord_123456",
        discord_id="123456",
        tracked_domains=["Distributed Systems"],
        profile_summary="Backend Engineer",
        similarity_threshold=0.80,
    )

    with patch.object(DiscordInteractionHandler, "get_or_create_user", new_callable=AsyncMock) as mock_get_user, \
         patch("app.bot.discord_client.AsyncSessionLocal"):
        mock_get_user.return_value = mock_user

        res = await DiscordInteractionHandler.handle_slash_command(
            name="research",
            options={"query": "foundations of theoretical cryptography"},
            user_discord_id="123456",
        )

        assert res.get("type") == 4
        content = res["data"]["content"]
        assert "Targeted Research Scan Initiated" in content
        assert "theoretical cryptography" in content
