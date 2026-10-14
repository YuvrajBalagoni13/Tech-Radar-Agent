#!/usr/bin/env python3
"""
Scrapes official category taxonomy from arXiv (https://arxiv.org/category_taxonomy)
and compiles a structured taxonomy JSON file for use in the planner node.
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scrape_arxiv_taxonomy")

ARXIV_TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "arxiv_taxonomy.json"

STOPWORDS: Set[str] = {
    "and", "or", "the", "in", "of", "for", "with", "on", "at", "by", "from", "to",
    "which", "that", "this", "these", "those", "such", "into", "an", "as", "is", "are",
    "covers", "including", "includes", "roughly", "material", "acm", "subject", "classes",
    "area", "areas", "have", "has", "having", "part", "particular", "some", "may", "also",
    "more", "likely", "primary", "separate", "although", "overlap", "description", "coming",
    "soon", "except", "other", "all", "both", "between", "well", "etc", "than", "been",
    "their", "then", "under", "over", "such", "only", "same", "very", "when", "where", "like",
    "algorithms", "algorithm", "analysis", "theory", "methods", "method", "applications",
    "approach", "problem", "problems", "results", "paper", "study", "studies", "framework",
    "performance", "systems", "system", "general", "wide", "view", "nature", "work", "various",
    "field", "fields", "topics", "topic", "research", "understanding", "related",
}

# Domain-specific enrichment for common engineering & research sub-fields
SPECIALIZED_ENRICHMENTS: Dict[str, Dict[str, Any]] = {
    "cs.CV": {
        "categories": ["cs.CV", "eess.IV"],
        "aliases": ["computer vision", "pattern recognition", "image generation", "vision models"],
        "terms": ["computer vision", "image generation", "diffusion models", "object detection", "vision transformer", "segmentation"],
        "github_topics": ["computer-vision", "image-generation", "diffusion-models", "vision-transformer", "generative-ai"],
        "anchors": ["vision", "image", "diffusion", "visual", "segmentation", "detection", "backbone", "convolution", "denoising", "pixel", "latent"],
    },
    "cs.CR": {
        "categories": ["cs.CR"],
        "aliases": ["cryptography", "cybersecurity", "security", "zero knowledge"],
        "terms": ["cryptography", "zero-knowledge", "zk-snark", "post-quantum", "homomorphic encryption", "cryptographic protocols"],
        "github_topics": ["cryptography", "zero-knowledge-proofs", "security", "post-quantum-cryptography", "cybersecurity"],
        "anchors": ["cryptography", "encryption", "cipher", "zero-knowledge", "snark", "stark", "homomorphic", "signature", "cryptographic", "zk"],
    },
    "cs.DC": {
        "categories": ["cs.DC", "cs.OS", "cs.NI"],
        "aliases": ["distributed systems", "parallel computing", "cluster computing", "cloud computing"],
        "terms": ["distributed systems", "consensus", "raft", "replication", "fault tolerance", "paxos"],
        "github_topics": ["distributed-systems", "raft-consensus", "fault-tolerance", "cluster-computing"],
        "anchors": ["distributed", "consensus", "raft", "paxos", "replication", "cluster", "fault-tolerant", "quorum"],
    },
    "cs.LG": {
        "categories": ["cs.LG", "cs.AI"],
        "aliases": ["machine learning", "deep learning", "representation learning", "reinforcement learning", "llm serving"],
        "terms": ["machine learning", "deep learning", "neural network", "reinforcement learning", "llm serving", "inference optimization"],
        "github_topics": ["machine-learning", "deep-learning", "pytorch", "reinforcement-learning", "llm-serving"],
        "anchors": ["learning", "training", "model", "neural", "deep", "loss", "dataset", "inference", "reinforcement", "policy"],
    },
    "cs.AI": {
        "categories": ["cs.AI", "cs.LG"],
        "aliases": ["artificial intelligence", "ai agents", "general ai"],
        "terms": ["artificial intelligence", "autonomous agents", "knowledge representation", "planning", "expert systems"],
        "github_topics": ["artificial-intelligence", "ai-agents", "autonomous-agents"],
        "anchors": ["intelligence", "agent", "planning", "knowledge", "reasoning", "heuristic", "autonomous"],
    },
    "cs.CL": {
        "categories": ["cs.CL"],
        "aliases": ["computational linguistics", "natural language processing", "nlp", "language models"],
        "terms": ["natural language processing", "nlp", "large language models", "transformers", "computational linguistics"],
        "github_topics": ["nlp", "transformers", "language-models", "llm"],
        "anchors": ["language", "linguistic", "transformer", "tokens", "syntax", "semantics", "generation", "prompt"],
    },
    "cs.DB": {
        "categories": ["cs.DB", "cs.IR"],
        "aliases": ["databases", "vector databases", "data management"],
        "terms": ["databases", "vector database", "query optimization", "indexing", "relational database", "hnsw"],
        "github_topics": ["database", "vector-database", "indexing", "hnsw"],
        "anchors": ["database", "indexing", "query", "transaction", "vector", "storage", "schema"],
    },
    "cs.AR": {
        "categories": ["cs.AR", "cs.DC"],
        "aliases": ["hardware architecture", "ai infrastructure", "accelerator design"],
        "terms": ["hardware architecture", "gpu architecture", "ai accelerator", "microarchitecture", "memory hierarchy"],
        "github_topics": ["computer-architecture", "hardware-acceleration", "fpga", "asic"],
        "anchors": ["hardware", "architecture", "gpu", "processor", "memory", "accelerator", "circuit", "silicon"],
    },
    "math.QA": {
        "categories": ["math.QA", "quant-ph"],
        "aliases": ["quantum algebra", "hopf algebra"],
        "terms": ["quantum algebra", "quantum groups", "hopf algebra", "braided categories", "knot invariants"],
        "github_topics": ["quantum-algebra", "mathematics", "algebraic-topology"],
        "anchors": ["quantum", "algebra", "hopf", "braided", "tensor", "representation", "qubit", "hamiltonian", "groups"],
    },
    "quant-ph": {
        "categories": ["quant-ph", "cs.ET"],
        "aliases": ["quantum physics", "quantum computing", "quantum information"],
        "terms": ["quantum computing", "quantum information", "qubit", "quantum error correction", "entanglement"],
        "github_topics": ["quantum-computing", "quantum-physics", "quantum-algorithms", "qiskit"],
        "anchors": ["quantum", "qubit", "entanglement", "circuit", "superposition", "decoherence", "teleportation"],
    },
}


def clean_tokens(text: str) -> List[str]:
    """Extract clean words without punctuation or stopwords."""
    words = re.findall(r"[a-zA-Z0-9_-]{3,}", text.lower())
    return [w for w in words if w not in STOPWORDS and not w.isdigit()]


def slugify(text: str) -> str:
    """Turn a phrase into a GitHub-compatible topic slug."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower().strip())
    return clean.strip("-")


def extract_terms(name: str, desc: str) -> List[str]:
    """Extract multi-word domain search terms from category title and description."""
    terms: List[str] = [name.lower()]

    # Extract comma or period-separated phrases from description
    phrases = re.split(r"[,;.]", desc)
    for p in phrases:
        p_clean = " ".join([w for w in re.findall(r"[a-zA-Z0-9_-]+", p.lower()) if w not in STOPWORDS])
        if 2 <= len(p_clean.split()) <= 4:
            terms.append(p_clean)

    # Add significant chunks of the name if conjunctive
    for part in re.split(r"\band\b|\bor\b|,", name.lower()):
        part = part.strip()
        if len(part) > 3 and part not in terms:
            terms.append(part)

    # Deduplicate preserving order
    deduped = []
    for t in terms:
        if t and t not in deduped:
            deduped.append(t)
    return deduped[:6]


def parse_taxonomy_html(html_or_soup: Any, source_url: str = ARXIV_TAXONOMY_URL) -> Dict[str, Any]:
    """Parse arXiv category taxonomy from HTML string or BeautifulSoup object."""
    if isinstance(html_or_soup, BeautifulSoup):
        soup = html_or_soup
    else:
        soup = BeautifulSoup(str(html_or_soup), "html.parser")

    categories_dict: Dict[str, Dict[str, Any]] = {}
    groups_dict: Dict[str, List[str]] = {}
    taxonomy_by_topic: Dict[str, Dict[str, Any]] = {}

    accordion_heads = soup.find_all("h2", class_="accordion-head")
    for h2 in accordion_heads:
        # Group name extraction
        button = h2.find("button")
        if button:
            group_name = button.get_text().strip()
        else:
            group_name = h2.get_text().strip()

        # Clean trailing labels like "Accordion panel" or "(Group ID)"
        group_name = re.sub(r"\s+", " ", group_name)
        group_name = re.sub(r"\s*\(.*?\)", "", group_name).strip()

        panel = h2.find_next_sibling("div")
        if not panel:
            continue

        if group_name not in groups_dict:
            groups_dict[group_name] = []

        div_items = panel.find_all("div", class_="columns divided")
        for div in div_items:
            h4 = div.find("h4")
            if not h4:
                continue

            code = h4.contents[0].strip() if h4.contents else ""
            span = h4.find("span")
            name = span.get_text().strip().strip("()") if span else ""

            # Skip header or uninformative rows
            if not code or not name or code.lower() in ("category name", "code"):
                continue

            desc_p = div.find("p")
            desc = desc_p.get_text().strip() if desc_p else ""

            # Check if this category has specialized enrichments
            enrichment = SPECIALIZED_ENRICHMENTS.get(code, {})

            if "terms" in enrichment:
                base_terms = list(enrichment["terms"])
            else:
                base_terms = extract_terms(name, desc)

            if "anchors" in enrichment:
                anchors_unique = list(enrichment["anchors"])
            else:
                base_anchors = clean_tokens(name) + clean_tokens(desc)
                anchors_unique = list(dict.fromkeys(base_anchors))[:15]

            base_github = [slugify(name)]
            for gh in enrichment.get("github_topics", []):
                if gh not in base_github:
                    base_github.append(gh)

            cat_info = {
                "code": code,
                "name": name,
                "group": group_name,
                "description": desc,
                "terms": base_terms,
                "github_topics": base_github,
                "anchors": anchors_unique,
            }
            categories_dict[code] = cat_info
            groups_dict[group_name].append(code)

            resolved_cats = enrichment.get("categories", [code])

            # Build topic entries for KNOWN_TAXONOMY
            # 1. Full name (lowercase)
            full_key = name.lower()
            topic_entry = {
                "name": name,
                "category_id": code,
                "categories": resolved_cats,
                "group": group_name,
                "terms": base_terms,
                "github_topics": base_github,
                "anchors": anchors_unique,
                "description": desc,
            }
            taxonomy_by_topic[full_key] = topic_entry

            # 2. Category code as a topic key
            taxonomy_by_topic[code.lower()] = topic_entry

            # 3. Conjunctive sub-names (e.g. "Cryptography and Security" -> "cryptography", "security")
            sub_parts = re.split(r"\band\b|\bor\b|,", full_key)
            for part in sub_parts:
                part = part.strip()
                if len(part) > 3 and part not in taxonomy_by_topic:
                    taxonomy_by_topic[part] = {
                        "name": name,
                        "category_id": code,
                        "categories": resolved_cats,
                        "group": group_name,
                        "terms": [part] + base_terms,
                        "github_topics": [slugify(part)] + base_github,
                        "anchors": anchors_unique,
                        "description": desc,
                    }

            # 4. Explicit aliases defined in enrichments
            for alias in enrichment.get("aliases", []):
                alias_lower = alias.lower()
                if alias_lower not in taxonomy_by_topic:
                    taxonomy_by_topic[alias_lower] = {
                        "name": name,
                        "category_id": code,
                        "categories": resolved_cats,
                        "group": group_name,
                        "terms": [alias_lower] + base_terms,
                        "github_topics": [slugify(alias_lower)] + base_github,
                        "anchors": anchors_unique,
                        "description": desc,
                    }

    # Add extra cross-cutting presets (e.g. LLM Serving, AI Infrastructure, Vector Databases)
    extra_presets = {
        "llm serving": {
            "name": "Large Language Model Serving & Inference Optimization",
            "category_id": "cs.LG",
            "categories": ["cs.LG", "cs.AI", "cs.CL"],
            "group": "Computer Science",
            "terms": ["llm serving", "vllm", "speculative decoding", "inference optimization", "kv cache"],
            "github_topics": ["llm-serving", "inference-engine", "vllm", "model-serving"],
            "anchors": ["serving", "inference", "throughput", "latency", "kv-cache", "decoding", "quantization"],
            "description": "High-performance LLM inference, kernel optimizations, and speculative decoding.",
        },
        "ai infrastructure": {
            "name": "AI Infrastructure & Distributed Training",
            "category_id": "cs.DC",
            "categories": ["cs.DC", "cs.LG", "cs.AR"],
            "group": "Computer Science",
            "terms": ["gpu training", "distributed training", "kernel optimization", "cuda", "model parallelism"],
            "github_topics": ["deep-learning-infrastructure", "cuda", "distributed-training"],
            "anchors": ["gpu", "cuda", "parallelism", "infrastructure", "kernel", "sharding", "interconnect"],
            "description": "High-throughput GPU/TPU training clusters, interconnects, and kernel scheduling.",
        },
        "vector databases": {
            "name": "Vector Databases & Approximate Nearest Neighbor Search",
            "category_id": "cs.DB",
            "categories": ["cs.DB", "cs.IR"],
            "group": "Computer Science",
            "terms": ["vector database", "hnsw", "approximate nearest neighbor", "vector search", "embedding index"],
            "github_topics": ["vector-database", "vector-search", "hnsw", "ann-search"],
            "anchors": ["vector", "embedding", "hnsw", "nearest-neighbor", "indexing", "similarity-search"],
            "description": "Dense vector retrieval, metric space search, and high-dimensional indexing.",
        },
        "image generation": {
            "name": "Image Generation & Latent Diffusion Models",
            "category_id": "cs.CV",
            "categories": ["cs.CV", "eess.IV"],
            "group": "Computer Science",
            "terms": ["image generation", "diffusion models", "generative adversarial", "text-to-image", "denoising diffusion"],
            "github_topics": ["image-generation", "diffusion-models", "generative-ai"],
            "anchors": ["image", "diffusion", "gan", "generation", "visual", "latent", "pixel", "denoising", "text-to-image"],
            "description": "Latent diffusion, score-based generative modeling, and text-to-image synthesis.",
        },
    }
    for k, v in extra_presets.items():
        taxonomy_by_topic[k] = v

    compiled_payload = {
        "metadata": {
            "source": source_url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "total_categories": len(categories_dict),
            "total_topics": len(taxonomy_by_topic),
            "total_groups": len(groups_dict),
        },
        "groups": groups_dict,
        "categories": categories_dict,
        "taxonomy": taxonomy_by_topic,
    }

    return compiled_payload


def scrape_arxiv_taxonomy(url: str = ARXIV_TAXONOMY_URL) -> Dict[str, Any]:
    """Scrape and parse the arXiv Category Taxonomy HTML page from web URL."""
    logger.info(f"Fetching arXiv taxonomy from {url}...")
    req = Request(url, headers={"User-Agent": "AutonomousTechRadarAgent/1.0 (+https://arxiv.org)"})
    with urlopen(req, timeout=15) as response:
        html = response.read().decode("utf-8")
    return parse_taxonomy_html(html, source_url=url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape official arXiv category taxonomy to JSON.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Target path for generated JSON taxonomy (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=ARXIV_TAXONOMY_URL,
        help="arXiv category taxonomy URL",
    )
    args = parser.parse_args()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = scrape_arxiv_taxonomy(args.url)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        meta = data["metadata"]
        logger.info(f"Successfully scraped and generated arXiv taxonomy!")
        logger.info(f"• Output File: {out_path}")
        logger.info(f"• Total Categories: {meta['total_categories']}")
        logger.info(f"• Total Topic Index Entries: {meta['total_topics']}")
        logger.info(f"• Total Groups: {meta['total_groups']}")
        print(f"\n[OK] arXiv taxonomy compiled to {out_path} ({meta['total_categories']} categories, {meta['total_topics']} topic mappings)")

    except Exception as e:
        logger.error(f"Failed to scrape and compile taxonomy: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
