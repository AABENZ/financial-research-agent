"""
Multi-agent orchestration for equity analysis
Uses CrewAI for Option A, designed to easily swap to LangGraph for Option B
"""

from .orchestrator import EquityAnalysisOrchestrator
from .sec_agent import SECFilingAgent
from .market_agent import MarketIntelligenceAgent
from .synthesis_agent import SynthesisAgent

__all__ = [
    "EquityAnalysisOrchestrator",
    "SECFilingAgent",
    "MarketIntelligenceAgent",
    "SynthesisAgent",
]
