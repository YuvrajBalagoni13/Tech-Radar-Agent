"""
Unit tests for the arXiv taxonomy scraper and planner node taxonomy loader.
"""

from pathlib import Path
from bs4 import BeautifulSoup
import pytest

from app.agent.nodes.planner_node import (
    KNOWN_TAXONOMY,
    TAXONOMY_FILE,
    _BUILTIN_TAXONOMY,
    load_arxiv_taxonomy,
)
from scripts.scrape_arxiv_taxonomy import (
    clean_tokens,
    extract_terms,
    parse_taxonomy_html,
    slugify,
)


def test_load_arxiv_taxonomy_from_json():
    """Verify that load_arxiv_taxonomy loads the scraped taxonomy JSON correctly."""
    assert TAXONOMY_FILE.is_file(), f"Taxonomy file {TAXONOMY_FILE} must exist"
    taxonomy = load_arxiv_taxonomy(TAXONOMY_FILE)

    # Should contain hundreds of topic mappings extracted from official arXiv categories
    assert len(taxonomy) >= 150

    # Check key domains exist
    assert "computer vision" in taxonomy
    assert "cryptography" in taxonomy
    assert "reinforcement learning" in taxonomy
    assert "quantum algebra" in taxonomy
    assert "distributed systems" in taxonomy

    # Verify structured attributes
    cv = taxonomy["computer vision"]
    assert "categories" in cv and "cs.CV" in cv["categories"]
    assert "terms" in cv and len(cv["terms"]) > 0
    assert "anchors" in cv and len(cv["anchors"]) > 0


def test_load_arxiv_taxonomy_fallback_on_missing_file():
    """Verify graceful fallback to built-in presets if the taxonomy file is missing or corrupted."""
    dummy_path = Path("/tmp/non_existent_arxiv_taxonomy_never_created.json")
    fallback_taxonomy = load_arxiv_taxonomy(dummy_path)

    assert fallback_taxonomy is not None
    assert len(fallback_taxonomy) >= len(_BUILTIN_TAXONOMY)
    for key in _BUILTIN_TAXONOMY:
        assert key in fallback_taxonomy


def test_known_taxonomy_global_availability():
    """Verify that KNOWN_TAXONOMY in planner_node is properly initialized."""
    assert isinstance(KNOWN_TAXONOMY, dict)
    assert len(KNOWN_TAXONOMY) > 0
    assert "computer vision" in KNOWN_TAXONOMY


def test_clean_tokens_and_slugify():
    """Verify clean_tokens excludes stopwords and slugify builds valid GitHub topics."""
    tokens = clean_tokens("Covers distributed systems, consensus algorithms like raft, and fault tolerance.")
    assert "distributed" in tokens
    assert "consensus" in tokens
    assert "raft" in tokens
    assert "fault" in tokens
    assert "and" not in tokens
    assert "like" not in tokens

    slug = slugify("Large Language Models & Optimization!")
    assert slug == "large-language-models-optimization"


def test_extract_terms():
    """Verify term extraction extracts title and descriptive phrases."""
    terms = extract_terms(
        "Distributed Systems",
        "Covers consensus protocols, fault tolerance, replication across clusters."
    )
    assert "distributed systems" in terms
    assert any("consensus" in t for t in terms)


def test_parse_arxiv_taxonomy_html_mock():
    """Test parsing a minimal mock HTML representation of the arXiv taxonomy page."""
    mock_html = """
    <html>
      <body>
        <div id="accordions">
          <h2 class="accordion-head"><button>Computer Science</button></h2>
          <div>
            <div class="columns divided">
              <div class="column">
                <h4>cs.AI <span>(Artificial Intelligence)</span></h4>
              </div>
              <div class="column">
                <p>Covers all aspects of AI including autonomous agents and reasoning.</p>
              </div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
    payload = parse_taxonomy_html(mock_html)
    categories = payload["categories"]
    taxonomy = payload["taxonomy"]

    assert "cs.AI" in categories
    assert categories["cs.AI"]["name"] == "Artificial Intelligence"
    assert categories["cs.AI"]["group"] == "Computer Science"
    assert "artificial intelligence" in taxonomy
    assert "cs.AI" in taxonomy["artificial intelligence"]["categories"]
    assert "reasoning" in taxonomy["artificial intelligence"]["anchors"]
