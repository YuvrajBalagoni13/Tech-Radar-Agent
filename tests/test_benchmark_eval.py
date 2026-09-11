"""
Integration & Regression Tests for Radar Benchmark Evaluation Suite.
Verifies Planner taxonomy accuracy, search mode discrimination,
cross-domain separation margins, false attribution rejection, and synthesis grounding.
"""

import pytest
from eval.benchmark_radar import (
    run_discovery_mode_benchmark,
    run_planner_taxonomy_benchmark,
    run_separation_margin_benchmark,
    run_synthesis_grounding_benchmark,
)


@pytest.mark.asyncio
async def test_benchmark_planner_taxonomy():
    """Verify Planner Node maps developer domains to arXiv categories with >= 90% accuracy."""
    res = await run_planner_taxonomy_benchmark()
    assert res["accuracy"] >= 90.0
    assert res["passed"] == res["total"]


@pytest.mark.asyncio
async def test_benchmark_mode_discrimination():
    """Verify query and sort discrimination between 'latest' and 'foundational' modes."""
    res = await run_discovery_mode_benchmark()
    assert res["mode_discrimination_passed"] is True


@pytest.mark.asyncio
async def test_benchmark_domain_separation_margin():
    """Verify in-domain vs cross-domain separation margin >= 0.35 and 0 false attributions."""
    res = await run_separation_margin_benchmark()
    assert res["avg_margin"] >= 0.35
    assert res["false_attributions"] == 0
    assert res["attribution_precision"] == 1.0


@pytest.mark.asyncio
async def test_benchmark_synthesis_grounding():
    """Verify synthesized tutorials contain zero hardcoded distributed DB hallucinations."""
    res = await run_synthesis_grounding_benchmark()
    assert res["grounding_score"] == 100.0
