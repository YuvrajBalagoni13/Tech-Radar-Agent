"""Tracks seen release content hashes in a JSON file to prevent duplicate alerts."""

import json
import logging
import os
from typing import Set

logger = logging.getLogger("techradar.dedup")

# Directory where persistent state is saved
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEDUP_FILE = os.path.join(DATA_DIR, "seen_hashes.json")


class DeduplicationStore:
    """In-memory and file-backed cache of processed release hashes."""

    _seen_hashes: Set[str] = set()
    _loaded: bool = False

    @classmethod
    def load(cls) -> Set[str]:
        """Load seen hashes from disk if not already in memory."""
        if not cls._loaded:
            try:
                if os.path.exists(DEDUP_FILE):
                    with open(DEDUP_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            cls._seen_hashes = set(data)
                        elif isinstance(data, dict):
                            cls._seen_hashes = set(data.keys())
                    logger.info(f"Loaded {len(cls._seen_hashes)} seen release hashes from {DEDUP_FILE}")
            except Exception as e:
                logger.warning(f"Failed to load dedup file {DEDUP_FILE}: {e}")
            cls._loaded = True
        return cls._seen_hashes

    @classmethod
    def is_seen(cls, content_hash: str) -> bool:
        """Check if a release content hash has already been processed."""
        cls.load()
        return content_hash in cls._seen_hashes

    @classmethod
    def mark_seen(cls, content_hash: str) -> None:
        """Mark a release content hash as seen and persist to disk."""
        cls.load()
        cls._seen_hashes.add(content_hash)
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(DEDUP_FILE, "w", encoding="utf-8") as f:
                json.dump(list(cls._seen_hashes), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist seen hashes to {DEDUP_FILE}: {e}")

    @classmethod
    def clear(cls) -> int:
        """Clear all recorded seen hashes."""
        cls.load()
        count = len(cls._seen_hashes)
        cls._seen_hashes.clear()
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(DEDUP_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            logger.info(f"Cleared {count} hashes from {DEDUP_FILE}")
        except Exception as e:
            logger.warning(f"Failed to clear {DEDUP_FILE}: {e}")
        return count

    @classmethod
    def count(cls) -> int:
        """Return total count of seen hashes."""
        cls.load()
        return len(cls._seen_hashes)
