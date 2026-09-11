"""
Targeted Ingestion Service for Tech Radar Agent.
Asynchronously polls HackerNews API, arXiv AI papers, and GitHub Trending/Releases.
Enforces SHA-256 cryptographic hashing to guarantee idempotent data ingestion.
"""

import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Optional
import httpx
from app.models.schemas import SearchPlan, TechReleaseItem

logger = logging.getLogger("techradar.ingestion")


class IngestionService:
    """
    Ingestion engine responsible for polling multi-source developer feeds,
    generating deterministic SHA-256 fingerprints, and normalizing raw metadata.
    """

    def __init__(self, timeout_seconds: float = 15.0):
        self.timeout = timeout_seconds
        self.headers = {
            "User-Agent": "AutonomousTechRadarAgent/1.0 (+https://github.com/enterprise/tech-radar-agent)"
        }

    @staticmethod
    def calculate_content_hash(source_url: str, title: str) -> str:
        """
        Generate a deterministic SHA-256 hash from canonical URL and normalized title.
        Ensures idempotent processing and eliminates duplicate alerts.
        """
        normalized_str = f"{source_url.strip().lower()}::{title.strip().lower()}"
        return hashlib.sha256(normalized_str.encode("utf-8")).hexdigest()

    async def fetch_hackernews_top(self, limit: int = 15, only_unseen: bool = True) -> List[TechReleaseItem]:
        """
        Ingest top technical stories from the official HackerNews Firebase REST API.
        If only_unseen=True, pages through candidate stories until `limit` unseen items
        are gathered or the safety candidate ceiling (100 IDs) is reached.
        """
        from app.services.dedup_store import DeduplicationStore

        items: List[TechReleaseItem] = []
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        max_candidates = min(max(limit * 5, 50), 100) if only_unseen else limit

        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                story_ids = resp.json()[:max_candidates]

                for sid in story_ids:
                    if len(items) >= limit:
                        break

                    item_url = f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
                    item_resp = await client.get(item_url)
                    if item_resp.status_code != 200:
                        continue

                    data = item_resp.json()
                    if not data or data.get("type") != "story":
                        continue

                    title = data.get("title", "").strip()
                    source_url = data.get("url") or f"https://news.ycombinator.com/item?id={sid}"
                    c_hash = self.calculate_content_hash(source_url, title)

                    if only_unseen and DeduplicationStore.is_seen(c_hash):
                        continue

                    score = data.get("score", 0)
                    summary = f"HackerNews discussion with {score} points and {data.get('descendants', 0)} comments: {title}"

                    items.append(
                        TechReleaseItem(
                            title=title,
                            source_url=source_url,
                            summary=summary,
                            source_type="hackernews",
                            published_at=datetime.fromtimestamp(data.get("time", 0), tz=timezone.utc),
                            content_hash=c_hash,
                            raw_payload=data,
                        )
                    )
        except Exception as e:
            logger.error(f"Error ingesting HackerNews top stories: {e}", exc_info=True)

        return items[:limit]

    async def fetch_arxiv_ai_papers(
        self,
        max_results: int = 10,
        only_unseen: bool = True,
        plan: Optional[SearchPlan] = None,
    ) -> List[TechReleaseItem]:
        """
        Query the official arXiv API for recent preprints.
        Dynamically adapts categories and query terms based on SearchPlan if provided;
        otherwise defaults to Artificial Intelligence, Computation & Language, ML, and SE.
        Supports discovery_mode: 'latest' (submittedDate) vs 'foundational' (relevance).
        If only_unseen=True, pages through up to max_pages (safety ceiling) until max_results
        unseen papers are collected.
        """
        from app.services.dedup_store import DeduplicationStore

        items: List[TechReleaseItem] = []
        base_url = "https://export.arxiv.org/api/query"
        max_pages = 4 if only_unseen else 1
        page_size = min(max(max_results * 2, 25), 50)

        # Dynamic category resolution from SearchPlan
        if plan and plan.arxiv_categories:
            valid_cats = [c.strip() for c in plan.arxiv_categories if c.strip()]
            cat_expr = " OR ".join([f"cat:{c}" for c in valid_cats]) if valid_cats else "cat:cs.AI"
        else:
            cat_expr = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.SE"

        is_foundational = bool(plan and plan.discovery_mode == "foundational")
        sort_by = "relevance" if is_foundational else "submittedDate"
        sort_order = "descending"

        # Terms expression
        terms_expr = ""
        if plan and plan.arxiv_query_terms:
            valid_terms = [t.strip() for t in plan.arxiv_query_terms if t.strip()]
            if valid_terms:
                term_clauses = [f'all:"{t}"' if " " in t else f"all:{t}" for t in valid_terms[:4]]
                terms_expr = " OR ".join(term_clauses)

        search_query = f"({cat_expr}) AND ({terms_expr})" if terms_expr else cat_expr

        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                for page in range(max_pages):
                    if len(items) >= max_results:
                        break

                    start_idx = page * page_size
                    params = {
                        "search_query": search_query,
                        "sortBy": sort_by,
                        "sortOrder": sort_order,
                        "start": str(start_idx),
                        "max_results": str(page_size),
                    }

                    resp = await client.get(base_url, params=params)
                    resp.raise_for_status()

                    # Parse Atom XML feed
                    root = ET.fromstring(resp.text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    entries = root.findall("atom:entry", ns)

                    # Fallback to category expression if strict terms query yielded 0 on page 0
                    if not entries and page == 0 and terms_expr:
                        logger.info(f"ArXiv query '{search_query}' returned 0 results. Retrying with broad '{cat_expr}'.")
                        params["search_query"] = cat_expr
                        resp = await client.get(base_url, params=params)
                        resp.raise_for_status()
                        root = ET.fromstring(resp.text)
                        entries = root.findall("atom:entry", ns)

                    if not entries:
                        break

                    for entry in entries:
                        if len(items) >= max_results:
                            break

                        title_elem = entry.find("atom:title", ns)
                        summary_elem = entry.find("atom:summary", ns)
                        id_elem = entry.find("atom:id", ns)
                        published_elem = entry.find("atom:published", ns)

                        if title_elem is None or summary_elem is None:
                            continue

                        title = " ".join(title_elem.text.strip().split())
                        summary = " ".join(summary_elem.text.strip().split())
                        source_url = id_elem.text.strip() if id_elem is not None else ""

                        c_hash = self.calculate_content_hash(source_url, title)

                        if only_unseen and DeduplicationStore.is_seen(c_hash):
                            continue

                        published_dt = None
                        if published_elem is not None and published_elem.text:
                            try:
                                published_dt = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
                            except Exception:
                                published_dt = datetime.now(timezone.utc)

                        items.append(
                            TechReleaseItem(
                                title=title,
                                source_url=source_url,
                                summary=summary,
                                source_type="arxiv",
                                published_at=published_dt,
                                content_hash=c_hash,
                                raw_payload={"arxiv_id": source_url},
                            )
                        )

                    # Respect arXiv API guidelines: courteous backoff between consecutive pages
                    if len(items) < max_results and page < max_pages - 1:
                        await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Error querying arXiv feed: {e}", exc_info=True)

        return items[:max_results]

    async def ingest_arxiv(
        self,
        limit: int = 15,
        plan: Optional[SearchPlan] = None,
        only_unseen: bool = False,
    ) -> List[TechReleaseItem]:
        """Convenience alias for fetch_arxiv_ai_papers."""
        return await self.fetch_arxiv_ai_papers(max_results=limit, plan=plan, only_unseen=only_unseen)

    async def fetch_github_trending(
        self,
        query: str = "language:python stars:>100 created:>2024-01-01",
        limit: int = 10,
        only_unseen: bool = True,
        plan: Optional[SearchPlan] = None,
    ) -> List[TechReleaseItem]:
        """
        Poll GitHub Search API for trending repositories and recent breakthrough toolkits.
        Adapts query and sorting based on SearchPlan if provided.
        If only_unseen=True, pages up to max_pages until limit unseen repos are retrieved.
        """
        from app.services.dedup_store import DeduplicationStore

        items: List[TechReleaseItem] = []
        api_url = "https://api.github.com/search/repositories"
        max_pages = 3 if only_unseen else 1
        per_page = min(max(limit * 2, 20), 30)

        effective_query = query
        sort_field = "updated"
        order_dir = "desc"

        if plan:
            terms = plan.github_topics or plan.arxiv_query_terms
            is_foundational = (plan.discovery_mode == "foundational")
            if is_foundational:
                sort_field = "stars"
                min_stars = 300
            else:
                sort_field = "updated"
                min_stars = 50

            if terms:
                joined_terms = " ".join([t.replace(" ", "-") for t in terms[:2]])
                effective_query = f"{joined_terms} stars:>{min_stars}"
            elif is_foundational:
                effective_query = "stars:>5000"

        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                for page in range(1, max_pages + 1):
                    if len(items) >= limit:
                        break

                    params = {
                        "q": effective_query,
                        "sort": sort_field,
                        "order": order_dir,
                        "page": page,
                        "per_page": per_page,
                    }

                    resp = await client.get(api_url, params=params)
                    if resp.status_code != 200:
                        break

                    data = resp.json()
                    repo_list = data.get("items", [])
                    if not repo_list:
                        break

                    for repo in repo_list:
                        if len(items) >= limit:
                            break

                        name = repo.get("full_name", "")
                        desc = repo.get("description") or "No description provided."
                        source_url = repo.get("html_url", "")
                        title = f"{name}: {repo.get('name', '')}"
                        c_hash = self.calculate_content_hash(source_url, title)

                        if only_unseen and DeduplicationStore.is_seen(c_hash):
                            continue

                        stars = repo.get("stargazers_count", 0)
                        summary = f"GitHub repository with {stars} stars: {desc}"

                        items.append(
                            TechReleaseItem(
                                title=title,
                                source_url=source_url,
                                summary=summary,
                                source_type="github",
                                published_at=datetime.now(timezone.utc),
                                content_hash=c_hash,
                                raw_payload=repo,
                            )
                        )
        except Exception as e:
            logger.error(f"Error fetching GitHub trending repositories: {e}", exc_info=True)

        return items[:limit]

    async def ingest_all(
        self,
        sources: Optional[List[str]] = None,
        limit_per_source: Optional[int] = None,
        only_unseen: bool = True,
        plan: Optional[SearchPlan] = None,
    ) -> List[TechReleaseItem]:
        """
        Aggregate technical publications across user-specified sources.
        Supported source identifiers: 'arxiv', 'hackernews', 'github'.
        Passes dynamic SearchPlan to arXiv and GitHub fetchers.
        If only_unseen=True, pages through feeds until target count of unseen items is met.
        Defaults to all sources if None provided.
        """
        active = [s.lower().strip() for s in (sources or ["hackernews", "arxiv", "github"])]
        lim = limit_per_source if limit_per_source is not None else 10

        total_items: List[TechReleaseItem] = []

        if any(s in active for s in ("hackernews", "hn")):
            hn_items = await self.fetch_hackernews_top(limit=lim, only_unseen=only_unseen)
            total_items.extend(hn_items)

        if any(s in active for s in ("arxiv", "papers", "research")):
            arxiv_items = await self.fetch_arxiv_ai_papers(max_results=lim, only_unseen=only_unseen, plan=plan)
            total_items.extend(arxiv_items)

        if any(s in active for s in ("github", "gh", "repos")):
            gh_lim = min(lim, 10)
            github_items = await self.fetch_github_trending(limit=gh_lim, only_unseen=only_unseen, plan=plan)
            total_items.extend(github_items)

        logger.info(
            f"Ingested {len(total_items)} fresh technical items across active sources {active} (target_per_source={lim}, only_unseen={only_unseen})."
        )
        return total_items
