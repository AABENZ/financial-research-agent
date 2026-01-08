"""
SEC Filing Analysis Agent
Analyzes SEC filings using the SEC analyzer tool
"""

from typing import Dict, Any, Optional
import logging

from .base import BaseAgent
from ..core.types import AnalysisRequest, SECFilingAnalysis
from ..tools.sec_analyzer import SECAnalyzer

logger = logging.getLogger(__name__)


class SECFilingAgent(BaseAgent):
    """
    Agent specialized in SEC filing analysis
    - Downloads and processes SEC filings
    - Performs component-based sentiment analysis
    - Identifies key risk factors and opportunities
    - Provides explainable insights
    """

    def __init__(self):
        super().__init__(
            name="SEC Filing Analyst",
            role="Expert in SEC filing analysis and regulatory document interpretation",
            goal="Extract deep insights from SEC filings with explainable sentiment analysis",
        )
        self.sec_analyzer = SECAnalyzer()

    async def execute(
        self, request: AnalysisRequest, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze SEC filing for the requested ticker

        Args:
            request: Analysis request with ticker and filing type
            context: Optional context (not used for this agent)

        Returns:
            Dictionary containing SEC analysis results
        """
        try:
            logger.info(
                f"SEC Agent analyzing {request.ticker} {request.filing_type}"
            )

            # Perform SEC analysis
            analysis = await self.sec_analyzer.analyze_filing(
                request.ticker, request.filing_type
            )

            if not analysis:
                logger.warning(f"No SEC analysis results for {request.ticker}")
                return {
                    "status": "failed",
                    "message": "Could not analyze SEC filing",
                }

            # Extract key insights
            insights = self._extract_insights(analysis)

            return {
                "status": "success",
                "analysis": analysis,
                "insights": insights,
                "summary": self._create_summary(analysis),
            }

        except Exception as e:
            logger.error(f"Error in SEC agent: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _extract_insights(self, analysis: SECFilingAnalysis) -> Dict[str, Any]:
        """Extract key insights from analysis"""
        insights = {
            "overall_sentiment": analysis.overall_sentiment.dominant,
            "confidence": analysis.overall_sentiment.confidence,
            "components": {},
            "key_risks": [],
            "key_opportunities": [],
        }

        # Analyze each component
        for comp_name, comp_analysis in analysis.components.items():
            insights["components"][comp_name] = {
                "sentiment": comp_analysis.sentiment.dominant,
                "confidence": comp_analysis.sentiment.confidence,
            }

            # Extract risks from risk_factors component
            if comp_name == "risk_factors":
                if comp_analysis.summary:
                    # Use actual sentences from the SEC filing
                    sentences = comp_analysis.summary.split('\n\n')
                    for sentence in sentences[:3]:
                        # Clean up bullet points and extract text
                        clean_text = sentence.replace('•', '').strip()
                        if len(clean_text) > 20:  # Meaningful text only
                            insights["key_risks"].append(
                                {
                                    "phrase": clean_text,
                                    "importance": 0.8,
                                }
                            )
                else:
                    # Fallback to generic if no summary available
                    insights["key_risks"].append({
                        "phrase": "Risk factors identified in SEC filing analysis",
                        "importance": 0.7,
                    })

            # Extract opportunities from strategy/financial components
            if comp_name in ["business_strategy", "financial_performance"]:
                for phrase in comp_analysis.key_phrases[:3]:
                    if phrase.sentiment == "positive":
                        insights["key_opportunities"].append(
                            {
                                "phrase": phrase.word,
                                "component": comp_name,
                            }
                        )

        return insights

    def _create_summary(self, analysis: SECFilingAnalysis) -> str:
        """Create human-readable summary"""
        summary = f"SEC Filing Analysis for {analysis.ticker} ({analysis.filing_type})\n\n"
        summary += f"Overall Sentiment: {analysis.overall_sentiment.dominant.upper()} "
        summary += f"(Confidence: {analysis.overall_sentiment.confidence:.2%})\n\n"

        summary += "Component Analysis:\n"
        for comp_name, comp_analysis in analysis.components.items():
            summary += f"- {comp_name.replace('_', ' ').title()}: "
            summary += f"{comp_analysis.sentiment.dominant.upper()} "
            summary += f"({comp_analysis.sentiment.confidence:.2%})\n"

        return summary
