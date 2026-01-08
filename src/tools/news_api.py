"""
News API tool for financial news gathering
Fetches recent news articles about companies/tickers
"""

from newsapi import NewsApiClient
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

from ..core.types import NewsArticle
from ..core.config import config

logger = logging.getLogger(__name__)


class NewsAPITool:
    """
    Fetches financial news using NewsAPI
    Provides recent articles for sentiment and trend analysis
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.market.news_api_key
        if not self.api_key:
            logger.warning("NewsAPI key not provided, news features will be limited")
            self.client = None
        else:
            self.client = NewsApiClient(api_key=self.api_key)

    def get_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days_back: int = 7,
        max_articles: int = 10,
    ) -> List[NewsArticle]:
        """
        Get recent news articles for a company

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better search
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return

        Returns:
            List of NewsArticle objects
        """
        if not self.client:
            logger.warning("NewsAPI client not initialized")
            return []

        try:
            logger.info(f"Fetching news for {ticker}, {days_back} days back")

            # Build search query
            search_query = f'({ticker}'
            if company_name:
                search_query += f' OR "{company_name}"'
            search_query += ') AND (stock OR finance OR earnings OR market)'

            # Date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Fetch articles
            response = self.client.get_everything(
                q=search_query,
                language="en",
                from_param=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                sort_by="relevancy",
                page_size=max_articles,
            )

            if response["status"] != "ok":
                logger.error(f"NewsAPI error: {response.get('message', 'Unknown error')}")
                return []

            # Convert to NewsArticle objects
            articles = []
            for article_data in response.get("articles", []):
                try:
                    article = NewsArticle(
                        title=article_data.get("title", "No title"),
                        source=article_data.get("source", {}).get("name", "Unknown"),
                        published_at=datetime.fromisoformat(
                            article_data.get("publishedAt", "").replace("Z", "+00:00")
                        ),
                        content=article_data.get("content") or article_data.get("description", ""),
                        url=article_data.get("url", ""),
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing article: {str(e)}")
                    continue

            logger.info(f"Retrieved {len(articles)} articles for {ticker}")
            return articles

        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []

    def get_news_summary(
        self, ticker: str, company_name: Optional[str] = None
    ) -> str:
        """
        Get a text summary of recent news

        Args:
            ticker: Stock ticker
            company_name: Company name

        Returns:
            Formatted string of news summaries
        """
        articles = self.get_news(ticker, company_name, days_back=7, max_articles=5)

        if not articles:
            return f"No recent news found for {ticker}"

        summary = f"Recent News for {ticker}:\n\n"
        for i, article in enumerate(articles, 1):
            summary += f"{i}. {article.title}\n"
            summary += f"   Source: {article.source} | {article.published_at.strftime('%Y-%m-%d')}\n"
            summary += f"   {article.content[:200]}...\n\n"

        return summary

    def get_sentiment_indicators(self, articles: List[NewsArticle]) -> Dict[str, int]:
        """
        Get basic sentiment indicators from article titles/content

        Args:
            articles: List of news articles

        Returns:
            Dictionary with positive/negative/neutral counts
        """
        # Simple keyword-based sentiment (can be enhanced with ML)
        positive_keywords = [
            "growth",
            "profit",
            "gain",
            "surge",
            "rally",
            "beat",
            "outperform",
            "success",
            "strong",
            "upgrade",
        ]
        negative_keywords = [
            "loss",
            "decline",
            "fall",
            "miss",
            "downgrade",
            "warning",
            "risk",
            "concern",
            "weak",
            "drop",
        ]

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        for article in articles:
            text = (article.title + " " + article.content).lower()

            pos_count = sum(1 for kw in positive_keywords if kw in text)
            neg_count = sum(1 for kw in negative_keywords if kw in text)

            if pos_count > neg_count:
                sentiment_counts["positive"] += 1
            elif neg_count > pos_count:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1

        return sentiment_counts
