"""Tests for ``python -m vecverdict`` module execution."""

from __future__ import annotations

import subprocess
import sys

from vecverdict import __version__


def test_module_execution_reports_version() -> None:
    """``python -m vecverdict --version`` must work without the console script."""
    result = subprocess.run(
        [sys.executable, "-m", "vecverdict", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout


def test_module_execution_shows_all_commands() -> None:
    """Module execution must expose the same subcommands as the console script."""
    result = subprocess.run(
        [sys.executable, "-m", "vecverdict", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for command in ("demo", "filter", "switch", "site"):
        assert command in result.stdout


def test_module_execution_propagates_failure() -> None:
    """A bad subcommand must exit non-zero so CI pipelines catch it."""
    result = subprocess.run(
        [sys.executable, "-m", "vecverdict", "nonexistent-command"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
