"""
Main orchestrator for equity analysis
Coordinates all agents to perform comprehensive analysis
"""

from typing import Dict, Any
import logging
import asyncio

from ..core.types import AnalysisRequest, EquityAnalysisResult
from .sec_agent import SECFilingAgent
from .market_agent import MarketIntelligenceAgent
from .synthesis_agent import SynthesisAgent

logger = logging.getLogger(__name__)


class EquityAnalysisOrchestrator:
    """
    Orchestrates multi-agent equity analysis
    Sequential execution: SEC → Market → Synthesis

    Design allows easy swap to parallel execution or different frameworks
    """

    def __init__(self):
        # Initialize all agents
        self.sec_agent = SECFilingAgent()
        self.market_agent = MarketIntelligenceAgent()
        self.synthesis_agent = SynthesisAgent()

        logger.info("Equity Analysis Orchestrator initialized")

    async def analyze(self, request: AnalysisRequest) -> EquityAnalysisResult:
        """
        Perform complete equity analysis

        Args:
            request: Analysis request specifying ticker and parameters

        Returns:
            Complete analysis result with recommendation
        """
        try:
            logger.info(f"Starting analysis for {request.ticker}")

            # Build context dictionary to pass between agents
            context = {}

            # Step 1: SEC Filing Analysis
            logger.info("Step 1: SEC Filing Analysis")
            sec_result = await self.sec_agent.execute(request, context)
            context["sec_agent"] = sec_result

            if sec_result.get("status") != "success":
                logger.warning(f"SEC analysis failed: {sec_result.get('message')}")

            # Step 2: Market Intelligence (can run in parallel with SEC in future)
            logger.info("Step 2: Market Intelligence")
            market_result = await self.market_agent.execute(request, context)
            context["market_agent"] = market_result

            if market_result.get("status") != "success":
                logger.warning(f"Market analysis failed: {market_result.get('message')}")

            # Step 3: Synthesis
            logger.info("Step 3: Synthesis & Recommendation")
            synthesis_result = await self.synthesis_agent.execute(request, context)

            if synthesis_result.get("status") != "success":
                raise Exception(f"Synthesis failed: {synthesis_result.get('message')}")

            # Extract final result
            analysis_result = synthesis_result["analysis_result"]

            logger.info(f"Analysis complete: {analysis_result.recommendation.sentiment}")

            return analysis_result

        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            raise

    async def analyze_batch(self, requests: list[AnalysisRequest]) -> list[EquityAnalysisResult]:
        """
        Analyze multiple tickers in parallel

        Args:
            requests: List of analysis requests

        Returns:
            List of analysis results
        """
        tasks = [self.analyze(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]

        return valid_results

    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        return {
            "sec_agent": "ready",
            "market_agent": "ready",
            "synthesis_agent": "ready",
        }
