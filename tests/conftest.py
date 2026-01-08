"""Pytest configuration and fixtures"""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ticker():
    """Sample ticker for testing"""
    return "TSLA"


@pytest.fixture
def sample_analysis_request():
    """Sample analysis request"""
    from src.core.types import AnalysisRequest

    return AnalysisRequest(
        ticker="TSLA",
        company_name="Tesla Inc",
        filing_type="10-K",
        include_news=False,  # Skip news for faster tests
        include_technicals=False,  # Skip technicals for faster tests
    )
