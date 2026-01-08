"""
Main SEC filing analyzer
Downloads, parses, and analyzes SEC filings with component-based sentiment analysis
"""

from sec_edgar_downloader import Downloader
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import asyncio

from ...core.types import (
    SECFilingAnalysis,
    ComponentAnalysis,
    SentimentScore,
    DocumentType,
)
from ...core.config import config
from ...models.sentiment import SentimentAnalyzer
from ...models.explainability import ExplainabilityEngine
from .component_analyzer import ComponentAnalyzer
from .text_extractor import TextExtractor

logger = logging.getLogger(__name__)


class SECAnalyzer:
    """
    Complete SEC filing analysis pipeline
    - Downloads filings from SEC EDGAR
    - Extracts text by component (Risk, Strategy, Financial, Operations)
    - Performs sentiment analysis with explainability
    - Returns structured results
    """

    def __init__(
        self,
        company_name: Optional[str] = None,
        email: Optional[str] = None,
    ):
        self.company_name = company_name or config.sec.company_name
        self.email = email or config.sec.email
        self.downloader = Downloader(self.company_name, self.email)

        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.explainability = ExplainabilityEngine(
            self.sentiment_analyzer.predict_proba
        )
        self.component_analyzer = ComponentAnalyzer()
        self.text_extractor = TextExtractor()

        # Cache directory
        self.cache_dir = Path(config.sec.cache_dir)

    def _get_filing_path(self, ticker: str, filing_type: str) -> Optional[str]:
        """
        Get path to downloaded filing

        Args:
            ticker: Stock ticker
            filing_type: Type of filing (10-K, 10-Q, etc.)

        Returns:
            Path to filing file or None if not found
        """
        try:
            base_path = self.cache_dir / ticker / filing_type

            if not base_path.exists():
                return None

            # Find the CIK directory (should be only one)
            subdirs = [d for d in base_path.iterdir() if d.is_dir()]
            if not subdirs:
                return None

            # Use the most recent filing
            cik_dir = subdirs[-1]

            # Look for the primary document or full submission
            for filename in ["primary-document.html", "full-submission.txt"]:
                file_path = cik_dir / filename
                if file_path.exists():
                    logger.info(f"Found filing at: {file_path}")
                    return str(file_path)

            return None

        except Exception as e:
            logger.error(f"Error getting filing path: {str(e)}")
            return None

    def _check_existing_filing(self, ticker: str, filing_type: str) -> bool:
        """Check if filing already exists locally"""
        return self._get_filing_path(ticker, filing_type) is not None

    async def download_filing(
        self, ticker: str, filing_type: str = "10-K", force: bool = False
    ) -> bool:
        """
        Download SEC filing

        Args:
            ticker: Stock ticker
            filing_type: Type of filing
            force: Force re-download even if exists

        Returns:
            True if successful
        """
        try:
            # Check if already exists
            if not force and self._check_existing_filing(ticker, filing_type):
                logger.info(f"Using cached filing for {ticker} {filing_type}")
                return True

            logger.info(f"Downloading {filing_type} for {ticker}...")

            # Download from past year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            self.downloader.get(
                filing_type,
                ticker,
                after=start_date.strftime("%Y-%m-%d"),
                before=end_date.strftime("%Y-%m-%d"),
                download_details=True,
            )

            # Verify download
            if self._get_filing_path(ticker, filing_type):
                logger.info(f"Successfully downloaded {ticker} {filing_type}")
                return True
            else:
                logger.warning(f"Download completed but filing not found")
                return False

        except Exception as e:
            logger.error(f"Error downloading filing: {str(e)}")
            return False

    async def analyze_filing(
        self, ticker: str, filing_type: str = "10-K"
    ) -> Optional[SECFilingAnalysis]:
        """
        Complete filing analysis pipeline

        Args:
            ticker: Stock ticker
            filing_type: Type of filing

        Returns:
            SECFilingAnalysis object or None if failed
        """
        try:
            # Set random seed for reproducibility
            from ...core.config import set_random_seed, config
            set_random_seed(config.model.random_seed)

            logger.info(f"Starting analysis for {ticker} {filing_type}")

            # Download filing
            if not await self.download_filing(ticker, filing_type):
                logger.error("Failed to download filing")
                return None

            # Get filing path
            file_path = self._get_filing_path(ticker, filing_type)
            if not file_path:
                logger.error("Could not find filing file")
                return None

            # Extract text by component
            text_by_component = self.text_extractor.extract_from_file(file_path)
            if not text_by_component:
                logger.error("No text extracted from filing")
                return None

            # Analyze each component
            component_analyses = {}
            all_sentiments = []

            for component_name, texts in text_by_component.items():
                logger.info(f"Analyzing {component_name}...")

                # Chunk texts for processing
                chunks = self.text_extractor.chunk_texts(texts, chunk_size=2048)

                if not chunks:
                    continue

                # Batch sentiment analysis
                sentiments = await self.sentiment_analyzer.analyze_batch(
                    chunks, DocumentType.SEC_FILING
                )

                # Calculate average sentiment for component
                avg_sentiment = SentimentScore(
                    positive=sum(s.positive for s in sentiments) / len(sentiments),
                    negative=sum(s.negative for s in sentiments) / len(sentiments),
                    neutral=sum(s.neutral for s in sentiments) / len(sentiments),
                )

                # Get explainability for significant chunks
                explanations = []
                risk_sentences = []  # Store actual text snippets

                for i, (chunk, sentiment) in enumerate(zip(chunks, sentiments)):
                    if sentiment.confidence > 0.6 and len(explanations) < 5:
                        # Get LIME word importance (use config for num_samples)
                        word_importance = self.explainability.explain(
                            chunk, num_features=8, num_samples=config.model.lime_num_samples
                        )
                        explanations.extend(word_importance)

                        # Extract actual sentences for risks (especially for risk_factors component)
                        if component_name == "risk_factors" and len(risk_sentences) < 5:
                            # Split chunk into sentences
                            sentences = [s.strip() for s in chunk.split('.') if len(s.strip()) > 50]
                            if sentences:
                                # Take first meaningful sentence
                                sentence_text = sentences[0][:300] + ('...' if len(sentences[0]) > 300 else '')

                                risk_sentences.append({
                                    'text': sentence_text,
                                    'importance': sentiment.confidence,
                                    'top_words': [w.word for w in word_importance[:3]]
                                })

                # Create summary from risk sentences for better display
                summary_text = None
                if component_name == "risk_factors" and risk_sentences:
                    summary_text = '\n\n'.join([f"• {r['text']}" for r in risk_sentences[:3]])

                # Store component analysis
                component_analyses[component_name] = ComponentAnalysis(
                    component_name=component_name,
                    sentiment=avg_sentiment,
                    key_phrases=explanations[:10],  # Top 10 LIME words (for debugging)
                    text_length=sum(len(t) for t in texts),
                    num_chunks=len(chunks),
                    summary=summary_text,  # Actual text snippets
                )

                all_sentiments.extend(sentiments)
                logger.info(f"Completed {component_name}: {avg_sentiment.dominant}")

            # Calculate overall sentiment
            if all_sentiments:
                overall_sentiment = SentimentScore(
                    positive=sum(s.positive for s in all_sentiments)
                    / len(all_sentiments),
                    negative=sum(s.negative for s in all_sentiments)
                    / len(all_sentiments),
                    neutral=sum(s.neutral for s in all_sentiments) / len(all_sentiments),
                )
            else:
                overall_sentiment = SentimentScore(
                    positive=0.0, negative=0.0, neutral=1.0
                )

            # Create result object
            result = SECFilingAnalysis(
                ticker=ticker,
                filing_type=filing_type,
                components=component_analyses,
                overall_sentiment=overall_sentiment,
            )

            logger.info(f"Analysis complete: Overall {overall_sentiment.dominant}")
            return result

        except Exception as e:
            logger.error(f"Error in analyze_filing: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    async def analyze_multiple_filings(
        self, ticker: str, filing_types: List[str] = ["10-K", "10-Q"]
    ) -> Dict[str, SECFilingAnalysis]:
        """
        Analyze multiple filing types for a ticker

        Args:
            ticker: Stock ticker
            filing_types: List of filing types to analyze

        Returns:
            Dictionary mapping filing types to analysis results
        """
        results = {}

        for filing_type in filing_types:
            result = await self.analyze_filing(ticker, filing_type)
            if result:
                results[filing_type] = result

        return results
