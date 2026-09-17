"""LangGraph workflow definition and state resumption helpers."""

import logging
from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes.compiler_node import compiler_node
from app.agent.nodes.domain_filter_node import domain_filter_node
from app.agent.nodes.escalation_node import escalation_node
from app.agent.nodes.notification_node import notification_node
from app.agent.nodes.research_node import research_node
from app.agent.nodes.synthesis_node import synthesis_node
from app.agent.state import AgentState
from app.core.config import settings

logger = logging.getLogger("techradar.graph")

# Fallback checkpointer for local/tests
_in_memory_checkpointer = MemorySaver()
_cached_graph = None


def route_filter(state: AgentState) -> str:
    """Route relevant papers to Discord/Telegram, or drop if irrelevant."""
    if state.get("is_relevant", False):
        primary = getattr(settings, "PRIMARY_NOTIFICATION_CHANNEL", "DISCORD").upper()
        user_id = str(state.get("user_id", ""))
        if primary == "TELEGRAM" or user_id.startswith("telegram_"):
            return "escalate_to_telegram"
        return "send_discord_alert"
    return END


def route_discord_decision(state: AgentState) -> str:
    """Check user decision from Discord (approve/skip/escalate)."""
    status = state.get("notification_status")
    if status == "USER_APPROVED":
        return "conduct_research"
    elif status in ("ESCALATED_TELEGRAM", "TIMEOUT"):
        return "escalate_to_telegram"
    elif status == "USER_IGNORED":
        return END
    # Default when interrupted at checkpoint
    return END


def route_telegram_decision(state: AgentState) -> str:
    """Check user decision from Telegram (approve/skip)."""
    status = state.get("notification_status")
    if status == "USER_APPROVED":
        return "conduct_research"
    elif status == "USER_IGNORED":
        return END
    return END


def create_radar_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> Any:
    """Build and compile the main radar state graph."""
    saver = checkpointer if checkpointer is not None else _in_memory_checkpointer

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("domain_filter", domain_filter_node)
    builder.add_node("send_discord_alert", notification_node)
    builder.add_node("escalate_to_telegram", escalation_node)
    builder.add_node("conduct_research", research_node)
    builder.add_node("synthesize_tutorial", synthesis_node)
    builder.add_node("compile_and_deliver_pdf", compiler_node)

    # Edges & routing
    builder.add_edge(START, "domain_filter")

    builder.add_conditional_edges(
        "domain_filter",
        route_filter,
        {
            "send_discord_alert": "send_discord_alert",
            "escalate_to_telegram": "escalate_to_telegram",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "send_discord_alert",
        route_discord_decision,
        {
            "conduct_research": "conduct_research",
            "escalate_to_telegram": "escalate_to_telegram",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "escalate_to_telegram",
        route_telegram_decision,
        {
            "conduct_research": "conduct_research",
            END: END,
        },
    )

    builder.add_edge("conduct_research", "synthesize_tutorial")
    builder.add_edge("synthesize_tutorial", "compile_and_deliver_pdf")
    builder.add_edge("compile_and_deliver_pdf", END)

    # Pause after sending alerts to wait for user click/command
    compiled_graph = builder.compile(
        checkpointer=saver,
        interrupt_after=["send_discord_alert", "escalate_to_telegram"],
    )

    logger.info("Compiled LangGraph StateGraph with human-in-the-loop checkpoints.")
    return compiled_graph


def get_radar_graph() -> Any:
    """Get or create singleton graph instance."""
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = create_radar_graph()
    return _cached_graph


async def resume_graph(
    thread_id: str,
    action: str,
    user_prompt_override: Optional[str] = None,
    graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """Resume a paused thread after user approves or skips."""
    active_graph = graph or get_radar_graph()
    config = {"configurable": {"thread_id": thread_id}}

    status_map = {
        "generate_tutorial": "USER_APPROVED",
        "skip_release": "USER_IGNORED",
        "escalate": "ESCALATED_TELEGRAM",
    }
    notification_status = status_map.get(action, "USER_IGNORED")

    update_payload: Dict[str, Any] = {
        "notification_status": notification_status,
    }
    if user_prompt_override:
        update_payload["user_prompt_override"] = user_prompt_override

    logger.info(f"Resuming LangGraph thread '{thread_id}' with action='{action}' (status={notification_status})")

    # Update state and resume execution
    await active_graph.aupdate_state(config, update_payload)

    final_state = await active_graph.ainvoke(None, config=config)
    logger.info(f"Thread '{thread_id}' execution completed. Notification status: {final_state.get('notification_status')}")
    return final_state
