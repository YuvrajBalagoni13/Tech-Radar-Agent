"""Executes search queries for a release item to retrieve context and benchmark trade-offs."""

import logging
from typing import Any, Dict, List
from app.agent.state import AgentState
from app.models.schemas import TechReleaseItem
from app.services.search import TechnicalSearchService

logger = logging.getLogger("techradar.research_node")


async def research_node(state: AgentState) -> Dict[str, Any]:
    """Run search queries scoped to matched domains and gather research notes."""
    release_raw = state.get("release_item")
    if isinstance(release_raw, dict):
        release = TechReleaseItem(**release_raw)
    else:
        release = release_raw

    matched_domains = state.get("matched_domains", [])
    user_prompt_override = state.get("user_prompt_override")

    search_service = TechnicalSearchService()

    # Primary query scoped to matched domains
    primary_query = search_service.construct_scoped_query(
        release_title=release.title,
        matched_domains=matched_domains,
        user_prompt_override=user_prompt_override,
    )

    # Secondary query targeting benchmarks and trade-offs
    domains_str = ", ".join(matched_domains) if matched_domains else "System Architecture"
    secondary_query = f"{release.title} architecture benchmarks trade-offs scalability limitations ({domains_str})"

    logger.info(f"Conducting deep technical research with query: '{primary_query}'")

    output_primary = await search_service.execute_search(primary_query, max_results=4)
    output_secondary = await search_service.execute_search(secondary_query, max_results=3)

    combined_sources = output_primary.sources + output_secondary.sources
    # Deduplicate sources by URL
    seen_urls = set()
    deduped_sources = []
    for s in combined_sources:
        url = s.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped_sources.append(s)

    research_notes: List[Dict[str, Any]] = [
        {
            "query": primary_query,
            "sources": deduped_sources,
            "findings": output_primary.synthesized_findings,
        },
        {
            "query": secondary_query,
            "sources": [],
            "findings": output_secondary.synthesized_findings,
        },
    ]

    return {
        "research_notes": research_notes,
    }
