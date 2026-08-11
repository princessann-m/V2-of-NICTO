"""Pytest configuration."""

import pytest

from mom.config import MoMConfig, LLMConfig, load_config


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


@pytest.fixture
def mom_config():
    return MoMConfig(
        llm=LLMConfig(provider="heuristic"),
        global_deadline=5.0,
        max_retries=1,
    )


@pytest.fixture
def orchestrator(mom_config):
    from mom.core.orchestrator import Orchestrator
    return Orchestrator(mom_config)
