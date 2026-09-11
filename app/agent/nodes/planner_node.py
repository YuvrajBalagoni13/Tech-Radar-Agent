"""
Search Strategy & Taxonomy Planner Node.
Intelligently compiles developer profile interests, tracked domains, and discovery mode
into a structured SearchPlan with exact arXiv taxonomy codes, GitHub topics, and grounding anchors.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
import httpx
from app.core.config import settings
from app.models.entities import UserProfile
from app.models.schemas import SearchPlan

logger = logging.getLogger("techradar.planner_node")

# Standard taxonomy dictionary for fast zero-cost local resolution
KNOWN_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "image generation": {
        "categories": ["cs.CV", "eess.IV"],
        "terms": ["image generation", "diffusion models", "generative adversarial", "text-to-image", "denoising diffusion"],
        "github_topics": ["image-generation", "diffusion-models", "generative-ai"],
        "anchors": ["image", "diffusion", "gan", "generation", "visual", "latent", "pixel", "denoising", "text-to-image"],
    },
    "computer vision": {
        "categories": ["cs.CV", "eess.IV"],
        "terms": ["computer vision", "object detection", "segmentation", "vision transformer"],
        "github_topics": ["computer-vision", "vision-transformer", "deep-learning"],
        "anchors": ["vision", "image", "visual", "segmentation", "detection", "backbone", "convolution"],
    },
    "cryptography": {
        "categories": ["cs.CR"],
        "terms": ["cryptography", "zero-knowledge", "zk-snark", "post-quantum", "homomorphic encryption"],
        "github_topics": ["cryptography", "zero-knowledge-proofs", "security", "post-quantum-cryptography"],
        "anchors": ["cryptography", "encryption", "cipher", "zero-knowledge", "snark", "stark", "homomorphic", "signature", "cryptographic"],
    },
    "quantum algebra": {
        "categories": ["math.QA", "quant-ph"],
        "terms": ["quantum algebra", "hopf algebra", "braided categories", "quantum groups", "lie algebra"],
        "github_topics": ["quantum-computing", "quantum-algebra", "mathematics"],
        "anchors": ["quantum", "algebra", "hopf", "braided", "tensor", "representation", "qubit", "hamiltonian"],
    },
    "quantum computing": {
        "categories": ["quant-ph", "cs.ET"],
        "terms": ["quantum computing", "qubit", "quantum circuit", "quantum error correction", "quantum algorithm"],
        "github_topics": ["quantum-computing", "quantum-circuits", "qiskit"],
        "anchors": ["quantum", "qubit", "circuit", "entanglement", "decoherence", "superposition"],
    },
    "distributed systems": {
        "categories": ["cs.DC", "cs.OS", "cs.NI"],
        "terms": ["distributed systems", "consensus", "raft", "replication", "fault tolerance"],
        "github_topics": ["distributed-systems", "raft-consensus", "fault-tolerance"],
        "anchors": ["distributed", "consensus", "raft", "paxos", "replication", "cluster", "fault-tolerant"],
    },
    "llm serving": {
        "categories": ["cs.LG", "cs.AI", "cs.CL"],
        "terms": ["llm serving", "vllm", "speculative decoding", "inference optimization", "kv cache"],
        "github_topics": ["llm-serving", "inference-engine", "vllm", "model-serving"],
        "anchors": ["serving", "inference", "throughput", "latency", "kv-cache", "decoding", "quantization"],
    },
    "ai infrastructure": {
        "categories": ["cs.DC", "cs.LG", "cs.AR"],
        "terms": ["gpu training", "distributed training", "kernel optimization", "cuda", "model parallelism"],
        "github_topics": ["deep-learning-infrastructure", "cuda", "distributed-training"],
        "anchors": ["gpu", "cuda", "parallelism", "infrastructure", "kernel", "sharding", "interconnect"],
    },
    "vector databases": {
        "categories": ["cs.DB", "cs.IR"],
        "terms": ["vector database", "hnsw", "approximate nearest neighbor", "vector search", "embedding index"],
        "github_topics": ["vector-database", "vector-search", "hnsw", "ann-search"],
        "anchors": ["vector", "embedding", "hnsw", "nearest-neighbor", "indexing", "similarity-search"],
    },
    "machine learning": {
        "categories": ["cs.LG", "cs.AI"],
        "terms": ["machine learning", "neural network", "deep learning", "gradient descent"],
        "github_topics": ["machine-learning", "deep-learning", "pytorch"],
        "anchors": ["learning", "training", "model", "neural", "deep", "loss", "dataset"],
    },
    "natural language processing": {
        "categories": ["cs.CL"],
        "terms": ["nlp", "language models", "transformer", "computational linguistics"],
        "github_topics": ["nlp", "transformers", "language-models"],
        "anchors": ["language", "tokens", "linguistic", "transformer", "syntax", "semantics"],
    },
    "theoretical computer science": {
        "categories": ["cs.CC", "cs.DS"],
        "terms": ["complexity theory", "algorithms", "computability", "automata"],
        "github_topics": ["algorithms", "theoretical-computer-science"],
        "anchors": ["complexity", "polynomial", "turing", "np-complete", "reduction", "graph"],
    },
    "reinforcement learning": {
        "categories": ["cs.LG", "cs.AI"],
        "terms": ["reinforcement learning", "policy gradient", "q-learning", "markov decision process"],
        "github_topics": ["reinforcement-learning", "rl", "deep-reinforcement-learning"],
        "anchors": ["reward", "policy", "agent", "markov", "reinforcement", "environment"],
    },
}


def _resolve_taxonomy_locally(
    tracked_domains: List[str],
    profile_summary: str,
    discovery_mode: str,
    ignored_keywords: List[str],
) -> SearchPlan:
    """Deterministic local taxonomy solver with multi-domain union and anchor extraction."""
    categories: set[str] = set()
    query_terms: list[str] = []
    github_topics: set[str] = set()
    anchors: set[str] = set()

    combined_text = f"{' '.join(tracked_domains)} {profile_summary}".lower()

    # Match against known taxonomies
    matched_any = False
    for key, spec in KNOWN_TAXONOMY.items():
        if key in combined_text or any(k in combined_text for k in key.split()):
            matched_any = True
            categories.update(spec["categories"])
            query_terms.extend(spec["terms"])
            github_topics.update(spec["github_topics"])
            anchors.update(spec["anchors"])

    # Fallback to direct extraction if unknown domain
    if not matched_any and tracked_domains:
        for d in tracked_domains:
            clean_d = d.strip()
            query_terms.append(clean_d)
            words = [w.lower() for w in re.findall(r"\w+", clean_d) if len(w) > 3]
            anchors.update(words)
            github_topics.add(clean_d.lower().replace(" ", "-"))
        categories.update(["cs.AI", "cs.LG", "cs.SE"])

    # Foundational mode tuning
    sort_by = "relevance" if discovery_mode == "foundational" else "submittedDate"
    if discovery_mode == "foundational":
        query_terms.extend(["foundational", "survey", "seminal", "principles"])

    # Ensure unique terms preserving order
    unique_terms = []
    for t in query_terms:
        if t not in unique_terms:
            unique_terms.append(t)

    gh_query = f"stars:>50 {' '.join(list(github_topics)[:2])}" if github_topics else "language:python stars:>100"

    return SearchPlan(
        user_id="default_user",
        discovery_mode=discovery_mode,
        arxiv_categories=sorted(list(categories)) or ["cs.AI"],
        arxiv_query_terms=unique_terms[:6],
        arxiv_sort_by=sort_by,
        arxiv_sort_order="descending",
        github_topics=sorted(list(github_topics))[:4],
        github_search_query=gh_query,
        anchor_concepts=sorted(list(anchors)),
        verification_anchor_concepts=sorted(list(anchors)),
        negative_filters=list(ignored_keywords),
    )


async def generate_search_plan(
    user: UserProfile,
    discovery_mode: str = "latest",
) -> SearchPlan:
    """
    Generate an actionable SearchPlan for multi-source ingestion and verification.
    Attempts LLM synthesis via Groq for high-nuance understanding, falling back
    instantly to high-coverage deterministic taxonomy resolution.
    """
    tracked = list(getattr(user, "tracked_domains", None) or [])
    summary = str(getattr(user, "profile_summary", "") or "").strip()
    ignored = list(getattr(user, "ignored_keywords", None) or [])
    user_id = str(getattr(user, "user_id", "default_user"))

    # Try fast Groq LPU reasoning if available
    if settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("gsk_your"):
        try:
            prompt = (
                f"You are a Senior Academic Search Strategist & Research Librarian.\n"
                f"A developer wants to search research papers on arXiv and code on GitHub.\n\n"
                f"User Profile & Interests: {summary or 'Software Engineer'}\n"
                f"Tracked Domains: {', '.join(tracked) if tracked else 'Computer Science'}\n"
                f"Discovery Mode: {discovery_mode} ('latest' = breaking papers, 'foundational' = seminal landmark papers)\n"
                f"Ignored Keywords: {', '.join(ignored)}\n\n"
                f"Generate a JSON SearchPlan with:\n"
                f"1. 'arxiv_categories': exact valid arXiv categories (e.g. 'cs.CV', 'cs.CR', 'math.QA', 'quant-ph', 'cs.LG', 'cs.DC', 'cs.SE')\n"
                f"2. 'arxiv_query_terms': 3-5 specific keyword phrases for title/abstract\n"
                f"3. 'github_topics': 2-3 GitHub repository topic tags\n"
                f"4. 'anchor_concepts': 5-10 technical words that MUST appear in candidate papers to prove domain relevance\n\n"
                f"Respond with strict JSON only. Example: {{\"arxiv_categories\": [\"cs.CR\"], \"arxiv_query_terms\": [\"zero knowledge\"], \"github_topics\": [\"cryptography\"], \"anchor_concepts\": [\"crypto\", \"cipher\", \"snark\"]}}"
            )

            endpoint = f"{settings.GROQ_API_BASE}/chat/completions"
            async with httpx.AsyncClient(timeout=8.0) as client:
                headers = {
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": settings.LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                }
                resp = await client.post(endpoint, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(data)
                    sort_by = "relevance" if discovery_mode == "foundational" else "submittedDate"
                    plan = SearchPlan(
                        user_id=user_id,
                        discovery_mode=discovery_mode,
                        arxiv_categories=parsed.get("arxiv_categories", ["cs.AI"]),
                        arxiv_query_terms=parsed.get("arxiv_query_terms", tracked[:3]),
                        arxiv_sort_by=sort_by,
                        arxiv_sort_order="descending",
                        github_topics=parsed.get("github_topics", []),
                        github_search_query=f"stars:>50 {' '.join(parsed.get('github_topics', []))}",
                        anchor_concepts=[a.lower() for a in parsed.get("anchor_concepts", [])],
                        verification_anchor_concepts=[a.lower() for a in parsed.get("anchor_concepts", [])],
                        negative_filters=ignored,
                    )
                    logger.info(f"Generated LLM SearchPlan for {user_id}: categories={plan.arxiv_categories}, mode={discovery_mode}")
                    return plan
        except Exception as e:
            logger.warning(f"Groq LLM planning failed: {e}. Falling back to deterministic taxonomy solver.")

    # High-coverage deterministic solver fallback
    plan = _resolve_taxonomy_locally(tracked, summary, discovery_mode, ignored)
    plan.user_id = user_id
    logger.info(f"Generated deterministic SearchPlan for {user_id}: categories={plan.arxiv_categories}, mode={discovery_mode}")
    return plan


async def plan_ad_hoc_research(
    query: str,
    user_id: str = "ad_hoc_user",
) -> SearchPlan:
    """
    Dynamically constructs an actionable SearchPlan for an ad-hoc natural language query.
    Detects user intent (foundational/thesis learning vs breaking latest work),
    maps topics to exact arXiv categories and query terms, and sets domain anchors.
    """
    clean_query = query.strip()
    query_lower = clean_query.lower()

    # Intent detection
    foundational_keywords = {
        "thesis", "learn", "learning", "study", "studying", "foundations",
        "foundational", "principles", "survey", "history", "seminal",
        "fundamental", "introduction", "basics", "textbook", "overview", "primer",
    }
    has_foundational = any(re.search(rf"\b{w}\b", query_lower) for w in foundational_keywords)
    has_latest = any(re.search(rf"\b{w}\b", query_lower) for w in ["latest", "recent", "new", "breakthrough", "yesterday", "current", "fresh"])

    discovery_mode = "latest" if (has_latest and not has_foundational) else "foundational" if has_foundational else "foundational"

    # Try fast Groq LPU reasoning if available
    if settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("gsk_your"):
        try:
            prompt = (
                f"You are a Senior Academic Search Strategist & Research Librarian.\n"
                f"A researcher submitted this natural language research request:\n"
                f"\"{clean_query}\"\n\n"
                f"Analyze the request and return a JSON SearchPlan with:\n"
                f"1. 'arxiv_categories': 1-3 exact valid arXiv categories (e.g. 'cs.CR' for cryptography/security, 'math.QA' for quantum algebra, 'quant-ph' for quantum physics, 'cs.CV' for computer vision/image generation, 'cs.LG' for machine learning, 'cs.DC' for distributed systems, 'cs.CL' for NLP, 'cs.AI')\n"
                f"2. 'arxiv_query_terms': 2-4 clean keyword search terms directly representing the research subject (e.g. ['theoretical cryptography', 'zero knowledge'])\n"
                f"3. 'discovery_mode': 'foundational' if the user is studying, writing a thesis, or learning, or 'latest' if looking for brand-new papers\n"
                f"4. 'anchor_concepts': 5-10 specific technical words that must appear in candidate papers to prove domain relevance\n\n"
                f"Respond with strict JSON only. Example: {{\"arxiv_categories\": [\"cs.CR\"], \"arxiv_query_terms\": [\"theoretical cryptography\", \"zero knowledge\"], \"discovery_mode\": \"foundational\", \"anchor_concepts\": [\"cryptography\", \"encryption\", \"cipher\", \"zk\", \"proof\"]}}"
            )

            endpoint = f"{settings.GROQ_API_BASE}/chat/completions"
            async with httpx.AsyncClient(timeout=8.0) as client:
                headers = {
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": settings.LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                }
                resp = await client.post(endpoint, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(data)
                    mode = parsed.get("discovery_mode", discovery_mode)
                    sort_by = "relevance" if mode == "foundational" else "submittedDate"
                    plan = SearchPlan(
                        user_id=user_id,
                        discovery_mode=mode,
                        arxiv_categories=parsed.get("arxiv_categories", ["cs.AI"]),
                        arxiv_query_terms=parsed.get("arxiv_query_terms", [clean_query]),
                        arxiv_sort_by=sort_by,
                        arxiv_sort_order="descending",
                        github_topics=parsed.get("github_topics", []),
                        github_search_query=clean_query,
                        anchor_concepts=[a.lower() for a in parsed.get("anchor_concepts", [])],
                        verification_anchor_concepts=[a.lower() for a in parsed.get("anchor_concepts", [])],
                        negative_filters=[],
                    )
                    logger.info(f"Generated LLM ad-hoc SearchPlan for '{clean_query[:50]}': categories={plan.arxiv_categories}, mode={plan.discovery_mode}")
                    return plan
        except Exception as e:
            logger.warning(f"Groq LLM ad-hoc planning failed: {e}. Falling back to deterministic solver.")

    # High-coverage deterministic solver fallback
    # Filter stopwords from query to identify core technical concepts
    stopwords = {
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "should", "now", "want", "need", "like", "coming",
        "comming", "thesis", "research", "learn", "accordingly", "papers", "paper",
        "study", "find", "get", "give", "show",
    }
    raw_words = re.findall(r"[a-zA-Z0-9_-]+", query_lower)
    core_words = [w for w in raw_words if w not in stopwords and len(w) > 2]
    candidate_concept = " ".join(core_words)

    categories: set[str] = set()
    query_terms: list[str] = []
    github_topics: set[str] = set()
    anchors: set[str] = set()

    # Match against KNOWN_TAXONOMY
    matched_any = False
    for key, spec in KNOWN_TAXONOMY.items():
        if key in query_lower or any(w in key.split() for w in core_words):
            matched_any = True
            categories.update(spec["categories"])
            query_terms.extend(spec["terms"])
            github_topics.update(spec["github_topics"])
            anchors.update(spec["anchors"])

    if not matched_any:
        if core_words:
            query_terms.append(candidate_concept)
            for w in core_words:
                anchors.add(w)
        categories.update(["cs.AI", "cs.LG"])

    # Ensure core keywords are present in query_terms at the beginning
    if candidate_concept and candidate_concept not in query_terms:
        query_terms.insert(0, candidate_concept)

    sort_by = "relevance" if discovery_mode == "foundational" else "submittedDate"

    # Deduplicate terms
    unique_terms = []
    for t in query_terms:
        if t not in unique_terms:
            unique_terms.append(t)

    plan = SearchPlan(
        user_id=user_id,
        discovery_mode=discovery_mode,
        arxiv_categories=sorted(list(categories)) or ["cs.AI"],
        arxiv_query_terms=unique_terms[:5],
        arxiv_sort_by=sort_by,
        arxiv_sort_order="descending",
        github_topics=sorted(list(github_topics))[:3],
        github_search_query=candidate_concept or clean_query,
        anchor_concepts=sorted(list(anchors)),
        verification_anchor_concepts=sorted(list(anchors)),
        negative_filters=[],
    )
    logger.info(f"Generated deterministic ad-hoc SearchPlan for '{clean_query[:50]}': categories={plan.arxiv_categories}, mode={plan.discovery_mode}")
    return plan
