"""
Synthesis Agent
Combines SEC filing analysis with market intelligence to generate final recommendation
Uses Gemini API for intelligent synthesis when available, falls back to rule-based logic
"""

from typing import Dict, Any, Optional, List
import logging
import os

from .base import BaseAgent
from ..core.types import (
    AnalysisRequest,
    InvestmentRecommendation,
    RiskFactor,
    EquityAnalysisResult,
)

logger = logging.getLogger(__name__)

# Import Gemini if available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini not available, using rule-based synthesis")


class SynthesisAgent(BaseAgent):
    """
    Agent that synthesizes all analysis into actionable recommendation
    - Cross-references SEC fundamentals with market action
    - Identifies discrepancies (e.g., positive filings but negative price action)
    - Generates evidence-based recommendation
    """

    def __init__(self):
        super().__init__(
            name="Chief Investment Strategist",
            role="Synthesize multi-source analysis into actionable investment insights",
            goal="Provide clear, evidence-based investment recommendations",
        )

        # Initialize Gemini if available
        self.use_gemini = False
        self.model = None

        if GEMINI_AVAILABLE:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    # Use Gemini 2.0 Flash (latest, fastest, has built-in caching)
                    self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    self.use_gemini = True
                    logger.info("Gemini 2.0 Flash initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {e}")
            else:
                logger.info("GEMINI_API_KEY not found, using rule-based synthesis")

    async def execute(
        self, request: AnalysisRequest, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize all analyses into final recommendation

        Args:
            request: Original analysis request
            context: Results from SEC and Market agents

        Returns:
            Final recommendation
        """
        try:
            logger.info(f"Synthesis Agent generating recommendation for {request.ticker}")

            if not context:
                return {
                    "status": "error",
                    "message": "No context provided for synthesis",
                }

            # Extract previous agent results
            sec_result = context.get("sec_agent", {})
            market_result = context.get("market_agent", {})

            # Generate recommendation
            recommendation = self._generate_recommendation(
                request, sec_result, market_result
            )

            # Create complete analysis result
            analysis_result = EquityAnalysisResult(
                request=request,
                sec_analysis=sec_result.get("analysis"),
                market_intelligence=market_result.get("intelligence"),
                recommendation=recommendation,
                metadata={
                    "sec_summary": sec_result.get("summary"),
                    "market_summary": market_result.get("summary"),
                },
            )

            return {
                "status": "success",
                "recommendation": recommendation,
                "analysis_result": analysis_result,
                "summary": self._create_summary(recommendation),
            }

        except Exception as e:
            import traceback
            logger.error(f"Error in Synthesis agent: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"status": "error", "message": str(e)}

    def _generate_recommendation(
        self,
        request: AnalysisRequest,
        sec_result: Dict,
        market_result: Dict,
    ) -> InvestmentRecommendation:
        """Generate investment recommendation based on all analyses"""

        # Extract key data
        sec_insights = sec_result.get("insights", {})
        market_insights = market_result.get("insights", {})

        # Use Gemini if available, otherwise fall back to rule-based
        if self.use_gemini:
            try:
                return self._generate_with_gemini(
                    request, sec_insights, market_insights
                )
            except Exception as e:
                logger.warning(f"Gemini synthesis failed, falling back to rules: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Fall through to rule-based logic

        # Rule-based logic (fallback or default)
        # Determine overall sentiment
        sentiment = self._determine_sentiment(sec_insights, market_insights)
        confidence = self._determine_confidence(sec_insights, market_insights)

        # Identify risks
        risks = self._identify_risks(sec_insights, market_insights)

        # Identify opportunities
        opportunities = self._identify_opportunities(sec_insights, market_insights)

        # Generate recommended action
        action = self._generate_action(sentiment, confidence, risks)

        # Build reasoning
        reasoning = self._build_reasoning(
            request.ticker, sentiment, sec_insights, market_insights, risks, opportunities
        )

        return InvestmentRecommendation(
            ticker=request.ticker,
            sentiment=sentiment,
            confidence=confidence,
            key_risks=risks,
            key_opportunities=opportunities,
            recommended_action=action,
            reasoning=reasoning,
        )

    def _generate_with_gemini(
        self,
        request: AnalysisRequest,
        sec_insights: Dict,
        market_insights: Dict,
    ) -> InvestmentRecommendation:
        """Generate recommendation using Gemini API"""

        # Build comprehensive prompt with all analysis data
        prompt = f"""You are a Chief Investment Strategist analyzing equity {request.ticker}.

**SEC Filing Analysis (FinBERT/SEC-BERT sentiment analysis):**
- Overall Sentiment: {sec_insights.get('overall_sentiment', 'N/A').upper()}
- Confidence: {sec_insights.get('confidence', 0):.2%}

Component Analysis:
"""

        # Add component sentiments
        components = sec_insights.get('components', {})
        if isinstance(components, dict):
            for comp, data in components.items():
                if isinstance(data, dict):
                    sentiment = data.get('sentiment', 'N/A')
                    confidence = data.get('confidence', 0)
                    prompt += f"- {comp.replace('_', ' ').title()}: {str(sentiment).upper()} ({confidence:.2%})\n"

        # Add identified risks from SEC filings
        prompt += "\nKey Risk Factors from SEC Filings:\n"
        key_risks = sec_insights.get('key_risks', [])
        if isinstance(key_risks, list):
            for risk in key_risks[:5]:
                if isinstance(risk, dict):
                    prompt += f"- {risk.get('phrase', 'N/A')}\n"
                elif isinstance(risk, str):
                    prompt += f"- {risk}\n"

        # Add opportunities from SEC filings
        prompt += "\nKey Opportunities from SEC Filings:\n"
        key_opps = sec_insights.get('key_opportunities', [])
        if isinstance(key_opps, list):
            for opp in key_opps[:5]:
                if isinstance(opp, dict):
                    prompt += f"- {opp.get('phrase', 'N/A')} ({opp.get('component', 'N/A')})\n"
                elif isinstance(opp, str):
                    prompt += f"- {opp}\n"

        # Add market intelligence
        prompt += f"""
**Market Intelligence:**
- Price Trend: {market_insights.get('price_trend', 'N/A')}
- News Sentiment: {market_insights.get('news_sentiment', 'N/A')}

Technical Signals:
"""
        tech_signals = market_insights.get('technical_signals', {})
        if isinstance(tech_signals, dict):
            for indicator, signal in tech_signals.items():
                prompt += f"- {indicator.upper()}: {signal}\n"

        # Add notable events
        notable_events = market_insights.get('notable_events', [])
        if isinstance(notable_events, list) and notable_events:
            prompt += "\nNotable News Events:\n"
            for event in notable_events[:3]:
                if isinstance(event, str):
                    prompt += f"- {event}\n"

        # Request structured output
        prompt += """
**Your Task:**
Synthesize the above fundamental analysis (SEC filings) and market intelligence into a comprehensive investment recommendation.

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, no extra text. Use spaces instead of tabs or special characters.

{
  "sentiment": "BULLISH or BEARISH or NEUTRAL",
  "confidence": "HIGH or MEDIUM or LOW",
  "risks": [
    {"category": "Fundamental or Technical or Sentiment", "description": "single line description", "severity": "high or medium or low", "evidence": ["brief evidence point"]}
  ],
  "opportunities": ["opportunity 1", "opportunity 2"],
  "action": "STRONG BUY or BUY or HOLD or SELL or AVOID - brief rationale",
  "reasoning": "2-3 paragraph analysis with no special formatting or line breaks within paragraphs"
}
"""

        # Call Gemini with JSON response mode
        logger.info(f"Calling Gemini 2.0 Flash for {request.ticker} synthesis...")

        # Configure for JSON output
        generation_config = {
            "response_mime_type": "application/json"
        }

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Parse response
        import json
        import re
        response_text = response.text.strip()

        # Log raw response for debugging
        logger.debug(f"Gemini raw response (first 500 chars): {response_text[:500]}")

        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Simple approach: Remove ALL newlines and tabs from JSON
        # JSON doesn't need pretty-printing to be valid
        response_text = response_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # Replace multiple spaces with single space
        response_text = re.sub(r'\s+', ' ', response_text)

        # Remove non-printable control characters (but we already removed \n, \r, \t)
        response_text = ''.join(
            char if char.isprintable() or char == ' ' else ' '
            for char in response_text
        )

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Log the problematic JSON for debugging
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Problematic JSON (first 1000 chars): {response_text[:1000]}")
            raise

        # Convert to InvestmentRecommendation
        risks = [
            RiskFactor(
                category=risk.get("category", "Unknown"),
                description=risk.get("description", ""),
                severity=risk.get("severity", "medium"),
                evidence=risk.get("evidence", []),
            )
            for risk in result.get("risks", [])
        ]

        return InvestmentRecommendation(
            ticker=request.ticker,
            sentiment=result.get("sentiment", "NEUTRAL"),
            confidence=result.get("confidence", "MEDIUM"),
            key_risks=risks,
            key_opportunities=result.get("opportunities", []),
            recommended_action=result.get("action", "HOLD - Insufficient data"),
            reasoning=result.get("reasoning", "No reasoning provided"),
        )

    def _determine_sentiment(
        self, sec_insights: Dict, market_insights: Dict
    ) -> str:
        """Determine overall sentiment (BULLISH/BEARISH/NEUTRAL)"""

        # SEC sentiment
        sec_sentiment = sec_insights.get("overall_sentiment", "neutral")

        # Market sentiment
        price_trend = market_insights.get("price_trend", "NEUTRAL")
        news_sentiment = market_insights.get("news_sentiment", "NEUTRAL")

        # Weighted decision
        bullish_signals = 0
        bearish_signals = 0

        # SEC analysis (higher weight)
        if sec_sentiment == "positive":
            bullish_signals += 2
        elif sec_sentiment == "negative":
            bearish_signals += 2

        # Price trend
        if price_trend == "BULLISH":
            bullish_signals += 1
        elif price_trend == "BEARISH":
            bearish_signals += 1

        # News sentiment
        if news_sentiment == "POSITIVE":
            bullish_signals += 1
        elif news_sentiment == "NEGATIVE":
            bearish_signals += 1

        # Determine overall
        if bullish_signals > bearish_signals + 1:
            return "BULLISH"
        elif bearish_signals > bullish_signals + 1:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _determine_confidence(
        self, sec_insights: Dict, market_insights: Dict
    ) -> str:
        """Determine confidence level (HIGH/MEDIUM/LOW)"""

        sec_confidence = sec_insights.get("confidence", 0)
        has_market_data = bool(market_insights.get("technical_signals"))
        has_news = bool(market_insights.get("notable_events"))

        # High confidence if SEC is confident and market confirms
        if sec_confidence > 0.7 and has_market_data and has_news:
            return "HIGH"
        # Low confidence if missing data or conflicting signals
        elif sec_confidence < 0.5 or not has_market_data:
            return "LOW"
        else:
            return "MEDIUM"

    def _identify_risks(
        self, sec_insights: Dict, market_insights: Dict
    ) -> List[RiskFactor]:
        """Identify key risk factors"""
        risks = []

        # SEC-identified risks
        sec_risks = sec_insights.get("key_risks", [])
        for risk in sec_risks[:3]:  # Top 3
            risks.append(
                RiskFactor(
                    category="Fundamental",
                    description=risk.get("phrase", "Unknown risk"),
                    severity="high" if risk.get("importance", 0) > 0.5 else "medium",
                    evidence=["SEC filing analysis"],
                )
            )

        # Market risks
        tech_signals = market_insights.get("technical_signals", {})
        if tech_signals.get("rsi") == "OVERBOUGHT":
            risks.append(
                RiskFactor(
                    category="Technical",
                    description="Stock appears overbought (RSI > 70)",
                    severity="medium",
                    evidence=["Technical indicators"],
                )
            )

        # News risks
        if market_insights.get("news_sentiment") == "NEGATIVE":
            risks.append(
                RiskFactor(
                    category="Sentiment",
                    description="Negative news sentiment detected",
                    severity="medium",
                    evidence=market_insights.get("notable_events", [])[:2],
                )
            )

        return risks

    def _identify_opportunities(
        self, sec_insights: Dict, market_insights: Dict
    ) -> List[str]:
        """Identify key opportunities"""
        opportunities = []

        # SEC opportunities
        sec_opps = sec_insights.get("key_opportunities", [])
        for opp in sec_opps[:3]:
            opportunities.append(f"{opp.get('phrase')} ({opp.get('component')})")

        # Technical opportunities
        tech_signals = market_insights.get("technical_signals", {})
        if tech_signals.get("rsi") == "OVERSOLD":
            opportunities.append("Potentially oversold - technical bounce opportunity")

        return opportunities

    def _generate_action(
        self, sentiment: str, confidence: str, risks: List[RiskFactor]
    ) -> str:
        """Generate recommended action"""

        high_severity_risks = sum(1 for r in risks if r.severity == "high")

        if sentiment == "BULLISH":
            if confidence == "HIGH" and high_severity_risks == 0:
                return "STRONG BUY - High conviction opportunity"
            elif confidence == "MEDIUM" or high_severity_risks <= 1:
                return "BUY - Favorable risk/reward, size position appropriately"
            else:
                return "HOLD - Monitor for risk reduction before entering"
        elif sentiment == "BEARISH":
            if confidence == "HIGH":
                return "SELL/AVOID - High confidence bearish outlook"
            else:
                return "HOLD - Bearish signals but insufficient confidence to sell"
        else:  # NEUTRAL
            return "HOLD - Insufficient conviction either way, await clearer signals"

    def _build_reasoning(
        self,
        ticker: str,
        sentiment: str,
        sec_insights: Dict,
        market_insights: Dict,
        risks: List[RiskFactor],
        opportunities: List[str],
    ) -> str:
        """Build detailed reasoning for recommendation"""

        reasoning = f"Analysis of {ticker}:\n\n"

        # SEC analysis
        reasoning += "Fundamental Analysis (SEC Filings):\n"
        reasoning += f"- Overall sentiment: {sec_insights.get('overall_sentiment', 'N/A').upper()}\n"
        reasoning += f"- Confidence: {sec_insights.get('confidence', 0):.2%}\n"

        comp_sentiments = sec_insights.get("components", {})
        if comp_sentiments:
            reasoning += "- Component analysis:\n"
            for comp, data in comp_sentiments.items():
                reasoning += f"  • {comp.replace('_', ' ').title()}: {data.get('sentiment', 'N/A').upper()}\n"

        # Market analysis
        reasoning += "\nMarket Analysis:\n"
        reasoning += f"- Price trend: {market_insights.get('price_trend', 'N/A')}\n"
        reasoning += f"- News sentiment: {market_insights.get('news_sentiment', 'N/A')}\n"

        tech_signals = market_insights.get("technical_signals", {})
        if tech_signals:
            reasoning += "- Technical signals:\n"
            for indicator, signal in tech_signals.items():
                reasoning += f"  • {indicator.upper()}: {signal}\n"

        # Synthesis
        reasoning += f"\nSynthesis: {sentiment} outlook"

        if risks:
            reasoning += f"\nKey risks identified: {len(risks)}"

        if opportunities:
            reasoning += f"\nKey opportunities: {', '.join(opportunities[:2])}"

        return reasoning

    def _create_summary(self, recommendation: InvestmentRecommendation) -> str:
        """Create concise summary"""
        summary = f"""
Investment Recommendation for {recommendation.ticker}

Sentiment: {recommendation.sentiment}
Confidence: {recommendation.confidence}

Key Risks:
{chr(10).join(f'- {risk.description}' for risk in recommendation.key_risks[:3])}

Recommended Action: {recommendation.recommended_action}
"""
        return summary.strip()
