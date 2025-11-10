"""
Pytest configuration for a1 tests.

Loads environment variables from .env file for API keys.
"""

import os
from pathlib import Path


def pytest_configure(config):
    """Load .env file before running tests."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
