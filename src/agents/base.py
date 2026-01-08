"""
Base agent interface for framework-agnostic design
Allows easy swap from CrewAI to LangGraph
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..core.types import AnalysisRequest


class BaseAgent(ABC):
    """
    Abstract base class for all agents
    Defines common interface regardless of framework
    """

    def __init__(self, name: str, role: str, goal: str):
        self.name = name
        self.role = role
        self.goal = goal

    @abstractmethod
    async def execute(self, request: AnalysisRequest, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent's task

        Args:
            request: Analysis request
            context: Optional context from previous agents

        Returns:
            Agent's output
        """
        pass

    def _format_output(self, data: Dict[str, Any]) -> str:
        """Format output for human readability"""
        return str(data)
