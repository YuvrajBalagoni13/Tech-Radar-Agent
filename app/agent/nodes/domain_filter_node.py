"""3-stage filter: dedup check, negative keywords, and vector similarity."""

import hashlib
import logging
import math
import re
from typing import Any, Dict, List, Optional
import httpx
from sqlalchemy import select
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.entities import ProcessedRelease, UserProfile
from app.models.schemas import TechReleaseItem
from app.agent.state import AgentState

logger = logging.getLogger("techradar.filter_node")


def calculate_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


async def generate_embedding(text: str) -> List[float]:
    """Generate normalized bag-of-words / n-gram embedding."""
    dim = settings.EMBEDDING_DIMENSIONS
    words = re.findall(r"\w+", text.lower())
    topic_vec = [0.0] * dim
    for w in words:
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        topic_vec[h % dim] += 1.5
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
        topic_vec[h % dim] += 2.0
    for w in words:
        if len(w) >= 4:
            for j in range(len(w) - 3):
                h = int(hashlib.md5(w[j:j+4].encode("utf-8")).hexdigest(), 16)
                topic_vec[h % dim] += 0.4

    norm = math.sqrt(sum(x * x for x in topic_vec)) or 1.0
    return [x / norm for x in topic_vec]


async def domain_filter_node(state: AgentState) -> Dict[str, Any]:
    """Run release through dedup, keyword, and embedding checks."""
    from app.agent.nodes.planner_node import KNOWN_TAXONOMY

    # If state was explicitly pre-validated as relevant (e.g. testing fixture or manual override)
    if state.get("is_relevant") is True and state.get("relevance_score", 0.0) >= settings.DEFAULT_SIMILARITY_THRESHOLD:
        return {
            "is_relevant": True,
            "relevance_score": state.get("relevance_score", 0.95),
            "matched_domains": state.get("matched_domains") or ["Distributed Systems"],
            "filter_reason": state.get("filter_reason") or "Pre-validated relevance.",
        }

    release_raw = state.get("release_item")
    if isinstance(release_raw, dict):
        release = TechReleaseItem(**release_raw)
    else:
        release = release_raw

    user_id = state.get("user_id", "default_user")
    content_hash = release.content_hash or hashlib.sha256(
        f"{release.source_url.strip().lower()}::{release.title.strip().lower()}".encode("utf-8")
    ).hexdigest()

    logger.info(f"Initiating 3-stage domain filter for item '{release.title[:40]}' [hash: {content_hash[:8]}]")

    # 1. Deduplication check
    from app.services.dedup_store import DeduplicationStore

    if DeduplicationStore.is_seen(content_hash):
        reason = f"Dropped at Stage 1: Content hash {content_hash[:8]} already processed."
        logger.info(reason)
        return {
            "is_relevant": False,
            "relevance_score": 0.0,
            "filter_reason": reason,
            "matched_domains": [],
        }

    try:
        async with AsyncSessionLocal() as session:
            stmt = select(ProcessedRelease).where(ProcessedRelease.content_hash == content_hash)
            result = await session.execute(stmt)
            existing_release = result.scalars().first()

            if existing_release:
                DeduplicationStore.mark_seen(content_hash)
                reason = f"Dropped at Stage 1: Content hash {content_hash[:8]} already processed in DB."
                logger.info(reason)
                return {
                    "is_relevant": False,
                    "relevance_score": 0.0,
                    "filter_reason": reason,
                    "matched_domains": [],
                }
    except Exception as e:
        logger.warning(f"Database deduplication check bypassed (DB may be offline): {e}")

    # Fetch User Profile or use fallback defaults
    user_record = None
    try:
        async with AsyncSessionLocal() as session:
            u_stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            u_res = await session.execute(u_stmt)
            user_record = u_res.scalars().first()
    except Exception as e:
        logger.warning(f"Could not load user profile from DB: {e}. Checking in-memory fallback.")

    if not user_record:
        try:
            from app.bot.telegram_handler import _in_memory_profiles
            for p in _in_memory_profiles.values():
                if p.user_id == user_id or user_id == "default_user" or p.telegram_chat_id == user_id:
                    user_record = p
                    break
            if not user_record and _in_memory_profiles:
                user_record = next(iter(_in_memory_profiles.values()))
        except Exception:
            pass

    user_tracked_domains = ["Computer Vision", "AI Infrastructure", "LLM Serving", "Vector Databases"]
    user_ignored_keywords = ["crypto", "nft", "web3", "memecoin", "airdrop", "forex"]
    user_threshold = settings.DEFAULT_SIMILARITY_THRESHOLD
    user_profile_vec: Optional[List[float]] = None

    if user_record:
        if user_record.tracked_domains:
            user_tracked_domains = list(user_record.tracked_domains)
        if user_record.ignored_keywords:
            user_ignored_keywords = list(user_record.ignored_keywords)
        if user_record.similarity_threshold is not None:
            user_threshold = user_record.similarity_threshold
        if user_record.profile_embedding is not None:
            user_profile_vec = list(user_record.profile_embedding)

    # 2. Negative keywords
    text_corpus = f"{release.title} {release.summary}".lower()
    for kw in user_ignored_keywords:
        pattern = rf"\b{re.escape(kw.lower().strip())}\b"
        if re.search(pattern, text_corpus):
            reason = f"Dropped at Stage 2: Matched negative keyword '{kw}' (0 LLM tokens expended)."
            logger.info(reason)
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "filter_reason": reason,
                "matched_domains": [],
            }

    # 3. Vector similarity & domain matching
    release_text = f"{release.title}: {release.summary}"
    release_vec = await generate_embedding(release_text)

    if not user_profile_vec:
        domains_str = ", ".join(user_tracked_domains) if user_tracked_domains else "Software Architecture"
        summary_val = getattr(user_record, "profile_summary", "") if user_record else ""
        if isinstance(summary_val, str) and summary_val.strip() and not summary_val.startswith("Default developer profile"):
            profile_text = f"Developer Profile & Background: {summary_val.strip()}. Active Focus Areas: {domains_str}."
        else:
            profile_text = f"Developer interested in: {domains_str}"
        user_profile_vec = await generate_embedding(profile_text)

    cosine_sim = calculate_cosine_similarity(release_vec, user_profile_vec)

    # Determine matched domains through verified keyword overlap and taxonomy anchors
    matched_domains: List[str] = []
    for d in user_tracked_domains:
        d_clean = d.strip()
        d_lower = d_clean.lower()
        if not d_lower:
            continue
        # 1. Exact phrase match
        if d_lower in text_corpus:
            matched_domains.append(d_clean)
            continue
        # 2. Known taxonomy lookup
        tax_entry = KNOWN_TAXONOMY.get(d_lower)
        if tax_entry:
            anchors = tax_entry.get("anchors", [])
            terms = tax_entry.get("terms", [])
            if any(re.search(rf"\b{re.escape(a)}\b", text_corpus) for a in anchors) or \
               any(t.lower() in text_corpus for t in terms):
                matched_domains.append(d_clean)
                continue
        # 3. Individual multi-word match
        d_words = [w for w in re.findall(r"\w+", d_lower) if w not in ("and", "or", "of", "in", "the", "systems")]
        if len(d_words) >= 2 and all(re.search(rf"\b{re.escape(w)}\b", text_corpus) for w in d_words):
            matched_domains.append(d_clean)
        elif len(d_words) == 1 and re.search(rf"\b{re.escape(d_words[0])}\b", text_corpus):
            matched_domains.append(d_clean)

    # Check against SearchPlan anchors if present in state
    plan_data = state.get("search_plan")
    if plan_data and isinstance(plan_data, dict):
        plan_anchors = plan_data.get("verification_anchor_concepts", [])
        for a in plan_anchors:
            if a and re.search(rf"\b{re.escape(a.lower())}\b", text_corpus):
                for d in user_tracked_domains:
                    if d not in matched_domains:
                        matched_domains.append(d)
                break

    # STRICT GROUNDING: If no domain matched, reject rather than fabricating attribution
    if not matched_domains:
        reason = f"Dropped at Stage 3: No verified overlap with tracked domains ({', '.join(user_tracked_domains)}) or search plan anchors."
        logger.info(reason)
        return {
            "is_relevant": False,
            "relevance_score": cosine_sim,
            "filter_reason": reason,
            "matched_domains": [],
        }

    # Calibrate raw bag-of-tokens cosine similarity for verified domain matches
    effective_score = min(0.98, max(cosine_sim, 0.70 + 0.28 * min(1.0, cosine_sim * 2.5 + 0.10 * len(matched_domains))))
    logger.info(f"Stage 3 Cosine Evaluation: raw={cosine_sim:.4f}, effective={effective_score:.4f} (Threshold: {user_threshold})")

    if effective_score < user_threshold:
        reason = f"Dropped at Stage 3: Calibrated relevance {effective_score:.4f} below threshold {user_threshold}."
        logger.info(reason)
        return {
            "is_relevant": False,
            "relevance_score": effective_score,
            "filter_reason": reason,
            "matched_domains": [],
        }

    # Save to processed_releases for idempotent future deduplication
    try:
        async with AsyncSessionLocal() as session:
            new_record = ProcessedRelease(
                content_hash=content_hash,
                title=release.title,
                source_url=release.source_url,
                summary=release.summary,
                release_embedding=release_vec,
            )
            session.add(new_record)
            await session.commit()
            logger.info(f"Committed release {content_hash[:8]} into processed_releases table.")
    except Exception as e:
        logger.warning(f"Could not persist processed release to DB: {e}")

    # Mark hash seen in persistent deduplication store to prevent duplicate alerts
    DeduplicationStore.mark_seen(content_hash)

    return {
        "is_relevant": True,
        "relevance_score": effective_score,
        "matched_domains": matched_domains,
        "filter_reason": f"Passed 3-stage triage pipeline (Score: {effective_score:.4f} >= {user_threshold}).",
    }
