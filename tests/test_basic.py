"""Basic sanity tests"""

import pytest
from src.core.config import config
from src.core.types import AnalysisRequest, SentimentScore


def test_config_loads():
    """Test that configuration loads properly"""
    assert config is not None
    assert config.model is not None
    assert config.sec is not None


def test_analysis_request_creation():
    """Test creating an analysis request"""
    request = AnalysisRequest(
        ticker="TSLA",
        company_name="Tesla Inc",
        filing_type="10-K",
    )

    assert request.ticker == "TSLA"
    assert request.company_name == "Tesla Inc"
    assert request.filing_type == "10-K"
    assert request.include_news is True  # default


def test_sentiment_score():
    """Test sentiment score calculations"""
    sentiment = SentimentScore(positive=0.7, negative=0.2, neutral=0.1)

    assert sentiment.dominant == "positive"
    assert sentiment.confidence == 0.7
    assert 0 <= sentiment.positive <= 1
    assert 0 <= sentiment.negative <= 1
    assert 0 <= sentiment.neutral <= 1


def test_sentiment_score_neutral():
    """Test neutral sentiment"""
    sentiment = SentimentScore(positive=0.3, negative=0.3, neutral=0.4)
    assert sentiment.dominant == "neutral"


@pytest.mark.parametrize(
    "ticker,expected",
    [
        ("TSLA", "TSLA"),
        ("aapl", "aapl"),
        ("RBLX", "RBLX"),
    ],
)
def test_ticker_handling(ticker, expected):
    """Test different ticker formats"""
    request = AnalysisRequest(ticker=ticker)
    assert request.ticker == expected
