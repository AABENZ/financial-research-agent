"""
Market Intelligence Agent
Gathers real-time market data and news
"""

from typing import Dict, Any, Optional
import logging
import os

from .base import BaseAgent
from ..core.types import AnalysisRequest, MarketIntelligence
from ..tools.market_data import MarketDataTool
from ..tools.news_api import NewsAPITool

logger = logging.getLogger(__name__)

# Import Gemini if available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini not available, using keyword-based news sentiment")


class MarketIntelligenceAgent(BaseAgent):
    """
    Agent specialized in market intelligence
    - Real-time price and technical analysis
    - News sentiment
    - Market trends and signals
    """

    def __init__(self):
        super().__init__(
            name="Market Intelligence Analyst",
            role="Expert in technical analysis, market trends, and news sentiment",
            goal="Provide real-time market intelligence and identify trading signals",
        )
        self.market_tool = MarketDataTool()
        self.news_tool = NewsAPITool()

        # Initialize Gemini for news sentiment if available
        self.use_gemini = False
        self.model = None

        if GEMINI_AVAILABLE:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    self.use_gemini = True
                    logger.info("Gemini 2.0 Flash initialized for news sentiment")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {e}")
            else:
                logger.info("GEMINI_API_KEY not found, using keyword-based sentiment")

    async def execute(
        self, request: AnalysisRequest, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gather market intelligence

        Args:
            request: Analysis request with ticker
            context: Optional context from previous agents

        Returns:
            Dictionary containing market intelligence
        """
        try:
            logger.info(f"Market Agent analyzing {request.ticker}")

            # Get market data
            market_data = None
            if request.include_technicals:
                market_data = self.market_tool.get_market_data(
                    request.ticker, period=request.period
                )

            # Get news
            news_articles = []
            if request.include_news:
                news_articles = self.news_tool.get_news(
                    request.ticker, request.company_name, days_back=7
                )

            # Create intelligence object
            intelligence = MarketIntelligence(
                ticker=request.ticker,
                market_data=market_data,
                news=news_articles,
            )

            # Generate insights
            insights = self._generate_insights(
                market_data, news_articles, request.ticker
            )

            return {
                "status": "success",
                "intelligence": intelligence,
                "insights": insights,
                "summary": self._create_summary(intelligence, insights),
            }

        except Exception as e:
            logger.error(f"Error in Market agent: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _generate_insights(
        self, market_data, news_articles, ticker: str
    ) -> Dict[str, Any]:
        """Generate market insights"""
        insights = {
            "price_trend": "UNKNOWN",
            "technical_signals": {},
            "news_sentiment": "NEUTRAL",
            "notable_events": [],
        }

        # Technical analysis
        if market_data:
            insights["price_trend"] = self.market_tool.get_price_trend(ticker)
            insights["technical_signals"] = self.market_tool.get_technical_signals(
                ticker
            )

        # News sentiment
        if news_articles:
            # Use Gemini for news sentiment if available
            if self.use_gemini:
                try:
                    insights["news_sentiment"] = self._analyze_news_with_gemini(
                        news_articles, ticker
                    )
                except Exception as e:
                    logger.warning(f"Gemini news analysis failed, using keywords: {e}")
                    # Fall back to keyword-based
                    insights["news_sentiment"] = self._analyze_news_with_keywords(
                        news_articles
                    )
            else:
                # Use keyword-based sentiment
                insights["news_sentiment"] = self._analyze_news_with_keywords(
                    news_articles
                )

            # Notable recent events
            insights["notable_events"] = [
                article.title for article in news_articles[:3]
            ]

        return insights

    def _analyze_news_with_gemini(self, news_articles, ticker: str) -> str:
        """Analyze news sentiment using Gemini"""

        # Build prompt with news articles
        prompt = f"""Analyze the overall sentiment of recent news articles about {ticker}.

**Recent News Articles:**
"""

        for i, article in enumerate(news_articles[:5], 1):
            prompt += f"\n{i}. **{article.title}**\n"
            prompt += f"   Source: {article.source}\n"
            prompt += f"   {article.content[:300]}...\n"

        prompt += """
**Your Task:**
Based on these articles, determine the overall news sentiment for this stock.

Respond with ONLY one word:
- POSITIVE (if news is generally favorable/bullish)
- NEGATIVE (if news is generally unfavorable/bearish)
- NEUTRAL (if mixed or no clear direction)
"""

        logger.info(f"Analyzing {len(news_articles)} news articles with Gemini...")
        response = self.model.generate_content(prompt)

        # Extract sentiment from response
        sentiment_text = response.text.strip().upper()

        # Validate response
        if "POSITIVE" in sentiment_text:
            return "POSITIVE"
        elif "NEGATIVE" in sentiment_text:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    def _analyze_news_with_keywords(self, news_articles) -> str:
        """Analyze news sentiment using keyword matching (fallback)"""
        sentiment_counts = self.news_tool.get_sentiment_indicators(news_articles)
        total = sum(sentiment_counts.values())

        if total > 0:
            if sentiment_counts["positive"] > sentiment_counts["negative"]:
                return "POSITIVE"
            elif sentiment_counts["negative"] > sentiment_counts["positive"]:
                return "NEGATIVE"

        return "NEUTRAL"

    def _create_summary(self, intelligence: MarketIntelligence, insights: Dict) -> str:
        """Create human-readable summary"""
        summary = f"Market Intelligence for {intelligence.ticker}\n\n"

        if intelligence.market_data:
            md = intelligence.market_data
            summary += f"Price: ${md.current_price:.2f} "
            summary += f"({md.price_change_pct:+.2f}%)\n"
            summary += f"Trend: {insights['price_trend']}\n"

            if md.rsi:
                summary += f"RSI: {md.rsi:.1f}\n"

        summary += f"\nNews Sentiment: {insights['news_sentiment']}\n"
        summary += f"Articles Analyzed: {len(intelligence.news)}\n"

        if insights["notable_events"]:
            summary += "\nRecent Events:\n"
            for event in insights["notable_events"]:
                summary += f"- {event}\n"

        return summary
