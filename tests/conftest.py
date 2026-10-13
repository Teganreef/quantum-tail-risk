"""Shared pytest fixtures. Tests never touch the network or real quantum hardware."""

import shutil
from pathlib import Path

import pytest

from qtr.config import DEFAULT_CONFIG_DIR


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    """A throwaway copy of config/ that a test can edit without touching the real files.

    The private portfolios.local.toml is NOT copied, so tests always see the public placeholders.
    """
    dest = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, dest, ignore=shutil.ignore_patterns("*.local.toml"))
    return dest
