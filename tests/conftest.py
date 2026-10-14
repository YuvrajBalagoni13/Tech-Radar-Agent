"""Pytest configuration and global test fixtures."""

import os
import pytest
from app.core.config import settings


@pytest.fixture(autouse=True, scope="session")
def configure_test_settings():
    """Ensure automated unit test suites use deterministic embedding to prevent API rate exhaustion."""
    original_provider = settings.EMBEDDING_PROVIDER
    settings.EMBEDDING_PROVIDER = "deterministic"
    os.environ["EMBEDDING_PROVIDER"] = "deterministic"
    yield
    settings.EMBEDDING_PROVIDER = original_provider
    os.environ["EMBEDDING_PROVIDER"] = original_provider
