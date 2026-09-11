"""
Integration Tests for LangGraph State Machine Execution Flow.
Tests end-to-end routing, human-in-the-loop checkpoint interruption,
and asynchronous state resumption.
"""

from unittest.mock import AsyncMock, patch
import pytest
from langgraph.checkpoint.memory import MemorySaver
from app.agent.graph import create_radar_graph, resume_graph
from app.agent.state import AgentState
from app.models.schemas import TechReleaseItem


@pytest.mark.asyncio
async def test_graph_flow_irrelevant_item_drops_to_end():
    """Verify that irrelevant item is filtered out and immediately terminates at END."""
    checkpointer = MemorySaver()
    graph = create_radar_graph(checkpointer=checkpointer)

    crypto_item = TechReleaseItem(
        title="NFT Airdrop Giveaway",
        source_url="https://spam.crypto/airdrop",
        summary="Airdrop and memecoin trading for profit.",
        source_type="hackernews",
    )

    initial_state: AgentState = {
        "release_item": crypto_item,
        "user_id": "test_dev_01",
        "relevance_score": 0.0,
        "is_relevant": False,
        "matched_domains": [],
        "filter_reason": "",
        "notification_status": "INITIALIZED",
        "user_prompt_override": None,
        "research_notes": [],
        "tutorial_markdown": None,
        "pdf_artifact_path": None,
        "error_logs": [],
    }

    thread_id = "thread-filter-test-1"
    config = {"configurable": {"thread_id": thread_id}}

    final_state = await graph.ainvoke(initial_state, config=config)

    assert final_state["is_relevant"] is False
    assert final_state["notification_status"] == "INITIALIZED"
    assert final_state.get("tutorial_markdown") is None
    assert final_state.get("pdf_artifact_path") is None


@pytest.mark.asyncio
async def test_graph_flow_approval_to_pdf_synthesis():
    """
    Verify complete human-in-the-loop workflow:
    1. Filter passes relevant item
    2. Dispatches Discord alert and interrupts
    3. User authorizes tutorial generation
    4. Resumes through research, synthesis, and PDF compilation
    """
    checkpointer = MemorySaver()
    graph = create_radar_graph(checkpointer=checkpointer)

    relevant_item = TechReleaseItem(
        title="Distributed Consensus via Raft: Architecture & Benchmarks",
        source_url="https://github.com/enterprise/distributed-consensus",
        summary="Distributed systems consensus algorithms for fault-tolerant vector databases.",
        source_type="github",
    )

    initial_state: AgentState = {
        "release_item": relevant_item,
        "user_id": "test_dev_02",
        "relevance_score": 0.95,
        "is_relevant": True,
        "matched_domains": ["Distributed Systems"],
        "filter_reason": "Manually authorized for testing",
        "notification_status": "INITIALIZED",
        "user_prompt_override": None,
        "research_notes": [],
        "tutorial_markdown": None,
        "pdf_artifact_path": None,
        "error_logs": [],
    }

    thread_id = "thread-approval-test-2"
    config = {"configurable": {"thread_id": thread_id}}

    from app.core.config import settings
    orig_channel = settings.PRIMARY_NOTIFICATION_CHANNEL
    settings.PRIMARY_NOTIFICATION_CHANNEL = "DISCORD"

    try:
        with patch("app.services.notifier.NotificationService.send_alert", new_callable=AsyncMock) as mock_alert, \
             patch("app.services.notifier.NotificationService.send_artifact", new_callable=AsyncMock) as mock_artifact:
            mock_alert.return_value = True
            mock_artifact.return_value = True

            # Phase 1: Run until interrupt after Discord alert
            paused_state = await graph.ainvoke(initial_state, config=config)
            assert paused_state["notification_status"] == "DISPATCHED_DISCORD"
            assert paused_state.get("tutorial_markdown") is None

            # Phase 2: Resume with user approval callback
            final_state = await resume_graph(
                thread_id=thread_id,
                action="generate_tutorial",
                user_prompt_override="Include memory leak profiling and ACID benchmarks",
                graph=graph,
            )

            assert final_state["notification_status"] == "USER_APPROVED"
            assert final_state.get("tutorial_markdown") is not None
            assert len(final_state.get("research_notes", [])) > 0
            assert final_state.get("pdf_artifact_path") is not None
            assert final_state["pdf_artifact_path"].endswith(".pdf")
    finally:
        settings.PRIMARY_NOTIFICATION_CHANNEL = orig_channel


@pytest.mark.asyncio
async def test_graph_flow_escalation_to_dismissal():
    """
    Verify timeout escalation flow:
    1. Discord alert times out -> escalates to Telegram
    2. Interrupts after Telegram alert
    3. User clicks skip / dismiss
    4. Safely terminates at END without generating tutorial
    """
    checkpointer = MemorySaver()
    graph = create_radar_graph(checkpointer=checkpointer)

    item = TechReleaseItem(
        title="Async Event Loops in Python: Latency Under Load",
        source_url="https://arxiv.org/abs/2409.0003",
        summary="High concurrency event loop mechanics and GIL contention.",
        source_type="arxiv",
    )

    initial_state: AgentState = {
        "release_item": item,
        "user_id": "test_dev_03",
        "relevance_score": 0.88,
        "is_relevant": True,
        "matched_domains": ["AI Infrastructure"],
        "filter_reason": "Passed threshold",
        "notification_status": "INITIALIZED",
        "user_prompt_override": None,
        "research_notes": [],
        "tutorial_markdown": None,
        "pdf_artifact_path": None,
        "error_logs": [],
    }

    thread_id = "thread-escalation-test-3"
    config = {"configurable": {"thread_id": thread_id}}

    from app.core.config import settings
    orig_channel = settings.PRIMARY_NOTIFICATION_CHANNEL
    settings.PRIMARY_NOTIFICATION_CHANNEL = "DISCORD"

    try:
        # Initial dispatch
        await graph.ainvoke(initial_state, config=config)

        # Simulate timeout trigger by resuming with escalate action
        escalated_state = await resume_graph(
            thread_id=thread_id,
            action="escalate",
            graph=graph,
        )
        assert escalated_state["notification_status"] == "ESCALATED_TELEGRAM"

        # User replies !skip
        closed_state = await resume_graph(
            thread_id=thread_id,
            action="skip_release",
            graph=graph,
        )
        assert closed_state["notification_status"] == "USER_IGNORED"
        assert closed_state.get("tutorial_markdown") is None
    finally:
        settings.PRIMARY_NOTIFICATION_CHANNEL = orig_channel


@pytest.mark.asyncio
async def test_graph_flow_telegram_primary_routing():
    """
    Verify that setting PRIMARY_NOTIFICATION_CHANNEL='TELEGRAM' routes
    alerts directly to Telegram with zero Discord dependency.
    """
    from app.core.config import settings
    orig_channel = settings.PRIMARY_NOTIFICATION_CHANNEL
    settings.PRIMARY_NOTIFICATION_CHANNEL = "TELEGRAM"

    try:
        checkpointer = MemorySaver()
        graph = create_radar_graph(checkpointer=checkpointer)

        item = TechReleaseItem(
            title="Llama 3.3 Architecture and Quantization",
            source_url="https://arxiv.org/abs/2409.0010",
            summary="Ultra-low latency inference optimizations on Groq LPUs.",
            source_type="arxiv",
        )

        initial_state: AgentState = {
            "release_item": item,
            "user_id": "test_dev_telegram_only",
            "relevance_score": 0.96,
            "is_relevant": True,
            "matched_domains": ["AI Infrastructure"],
            "filter_reason": "High relevance",
            "notification_status": "INITIALIZED",
            "user_prompt_override": None,
            "research_notes": [],
            "tutorial_markdown": None,
            "pdf_artifact_path": None,
            "error_logs": [],
        }

        thread_id = "thread-tg-primary-test-4"
        config = {"configurable": {"thread_id": thread_id}}

        with patch("app.services.notifier.NotificationService.send_alert", new_callable=AsyncMock) as mock_alert, \
             patch("app.services.notifier.NotificationService.send_artifact", new_callable=AsyncMock) as mock_artifact:
            mock_alert.return_value = True
            mock_artifact.return_value = True

            # Phase 1: Dispatches directly to Telegram and pauses at checkpoint
            paused_state = await graph.ainvoke(initial_state, config=config)
            assert paused_state["notification_status"] == "ESCALATED_TELEGRAM"

            # Phase 2: Resumes directly via Telegram approval
            final_state = await resume_graph(
                thread_id=thread_id,
                action="generate_tutorial",
                graph=graph,
            )
            assert final_state["notification_status"] == "USER_APPROVED"
            assert final_state.get("pdf_artifact_path") is not None
            assert final_state["pdf_artifact_path"].endswith(".pdf")
    finally:
        settings.PRIMARY_NOTIFICATION_CHANNEL = orig_channel
