"""
Type definitions and data models for Financial Research Agent
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentType(Enum):
    """Types of financial documents"""
    SEC_FILING = "sec_filing"
    EARNINGS_CALL = "earnings_call"
    NEWS_ARTICLE = "news_article"
    ANALYST_REPORT = "analyst_report"


class FilingType(Enum):
    """SEC filing types"""
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    PROXY = "DEF 14A"


class SentimentScore(BaseModel):
    """Sentiment analysis results"""
    positive: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)

    @property
    def dominant(self) -> str:
        """Get dominant sentiment"""
        scores = {"positive": self.positive, "negative": self.negative, "neutral": self.neutral}
        return max(scores, key=scores.get)

    @property
    def confidence(self) -> float:
        """Get confidence of dominant sentiment"""
        return max(self.positive, self.negative, self.neutral)


class WordImportance(BaseModel):
    """LIME word importance for explainability"""
    word: str
    importance: float
    sentiment: str
    magnitude: float


class ComponentAnalysis(BaseModel):
    """Analysis results for a filing component"""
    component_name: str
    sentiment: SentimentScore
    key_phrases: List[WordImportance] = []
    text_length: int
    num_chunks: int
    summary: Optional[str] = None


class SECFilingAnalysis(BaseModel):
    """Complete SEC filing analysis results"""
    ticker: str
    filing_type: str
    filing_date: Optional[datetime] = None
    components: Dict[str, ComponentAnalysis]
    overall_sentiment: SentimentScore
    timestamp: datetime = Field(default_factory=datetime.now)


class MarketData(BaseModel):
    """Market data snapshot"""
    ticker: str
    current_price: float
    price_change: float
    price_change_pct: float
    volume: int
    rsi: Optional[float] = None
    macd: Optional[Dict[str, float]] = None
    moving_averages: Optional[Dict[str, float]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class NewsArticle(BaseModel):
    """News article data"""
    title: str
    source: str
    published_at: datetime
    content: str
    url: str
    sentiment: Optional[SentimentScore] = None


class MarketIntelligence(BaseModel):
    """Market intelligence gathering results"""
    ticker: str
    market_data: Optional[MarketData] = None
    news: List[NewsArticle]
    analyst_sentiment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AnalysisRequest(BaseModel):
    """Request for equity analysis"""
    ticker: str
    company_name: Optional[str] = None
    filing_type: str = "10-K"
    include_news: bool = True
    include_technicals: bool = True
    period: str = "1mo"


class RiskFactor(BaseModel):
    """Identified risk factor"""
    category: str
    description: str
    severity: str  # "high", "medium", "low"
    evidence: List[str] = []


class InvestmentRecommendation(BaseModel):
    """Final investment recommendation"""
    ticker: str
    sentiment: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    key_risks: List[RiskFactor]
    key_opportunities: List[str]
    recommended_action: str
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class EquityAnalysisResult(BaseModel):
    """Complete equity analysis result"""
    request: AnalysisRequest
    sec_analysis: Optional[SECFilingAnalysis] = None
    market_intelligence: Optional[MarketIntelligence] = None
    recommendation: InvestmentRecommendation
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)
