#!/usr/bin/env python3
"""CLI utility to compare embeddings from Deterministic and Voyage AI (voyage-4-lite) embedders."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.embedding import compare_embeddings, compare_embeddings_batch


SAMPLE_PAIRS: List[Tuple[str, str, str]] = [
    (
        "Direct Topic Match (Cryptography)",
        "Zero-Knowledge SNARKs and recursive cryptographic proof composition",
        "Recursive zk-SNARK protocols for scalable verifiable computations",
    ),
    (
        "Semantic Cross-Domain Match (Distributed Consensus)",
        "Raft consensus algorithm with replicated state machine and leader leases",
        "Distributed database consistency models and Paxos quorum replication",
    ),
    (
        "Unrelated Topics (Negative Pair)",
        "Attention Is All You Need: Transformer architecture for sequence transduction",
        "Sourdough fermentation kinetics and high hydration bread baking techniques",
    ),
]


def print_comparison_result(result: dict, title: str = "") -> None:
    if title:
        print(f"\n==================== {title} ====================")
    print(f"Text A: {result['text_a']}")
    print(f"Text B: {result['text_b']}")
    print("-" * 60)
    print(f"Deterministic Similarity: {result['deterministic']['similarity']:.4f}  (dim={result['deterministic']['dimensions']})")
    if "voyage" in result and result["voyage"].get("similarity") is not None:
        print(f"Voyage AI ({result['voyage']['model']}): {result['voyage']['similarity']:.4f}  (dim={result['voyage']['dimensions']})")
    if "gemini" in result and result["gemini"].get("similarity") is not None:
        print(f"Google Gemini ({result['gemini']['model']}): {result['gemini']['similarity']:.4f}  (dim={result['gemini']['dimensions']})")
    print(f"Absolute Delta (Det vs Voyage): {result['absolute_difference']:.4f}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Deterministic vs. Voyage AI embeddings.")
    parser.add_argument("--text-a", type=str, help="First text string")
    parser.add_argument("--text-b", type=str, help="Second text string")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    args = parser.parse_args()

    if args.text_a and args.text_b:
        res = await compare_embeddings(args.text_a, args.text_b)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print_comparison_result(res, "Custom Query Comparison")
    else:
        print("Comparing Deterministic vs Voyage AI (voyage-4-lite) in a single batch call:")
        pairs = [(t1, t2) for _, t1, t2 in SAMPLE_PAIRS]
        labels = [title for title, _, _ in SAMPLE_PAIRS]
        results = await compare_embeddings_batch(pairs, labels=labels)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for item in results:
                print_comparison_result(item, item.get("label", ""))


if __name__ == "__main__":
    asyncio.run(main())
