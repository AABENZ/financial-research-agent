"""
Component-based analysis of SEC filings
Identifies and analyzes key sections: Risk, Strategy, Financial Performance, Operations
"""

from typing import Dict, List


class ComponentAnalyzer:
    """
    Identifies and categorizes different components of SEC filings
    Each component has specific keywords for identification
    """

    def __init__(self):
        self.components = {
            "financial_performance": {
                "keywords": [
                    "revenue",
                    "net income",
                    "earnings",
                    "profit",
                    "loss",
                    "cash flow",
                    "operating income",
                    "EBITDA",
                    "gross margin",
                    "operating margin",
                    "financial results",
                    "fiscal year",
                    "quarter",
                    "YoY",
                    "year-over-year",
                ],
                "weight": 1.0,
            },
            "risk_factors": {
                "keywords": [
                    "risk",
                    "uncertainty",
                    "challenge",
                    "threat",
                    "adverse",
                    "volatile",
                    "fluctuation",
                    "litigation",
                    "regulatory",
                    "competition",
                    "competitive pressure",
                    "market condition",
                    "economic condition",
                    "material adverse effect",
                ],
                "weight": 1.2,  # Higher weight for risk analysis
            },
            "business_strategy": {
                "keywords": [
                    "strategy",
                    "strategic",
                    "initiative",
                    "growth",
                    "expansion",
                    "acquisition",
                    "partnership",
                    "innovation",
                    "competitive advantage",
                    "market opportunity",
                    "business model",
                    "long-term",
                    "investment",
                    "R&D",
                    "research and development",
                ],
                "weight": 1.0,
            },
            "operations": {
                "keywords": [
                    "operations",
                    "operational",
                    "production",
                    "capacity",
                    "efficiency",
                    "supply chain",
                    "customers",
                    "users",
                    "daily active users",
                    "engagement",
                    "platform",
                    "infrastructure",
                    "employee",
                    "workforce",
                ],
                "weight": 0.9,
            },
        }

    def identify_component(self, text: str) -> List[str]:
        """
        Identify which components a text snippet belongs to

        Args:
            text: Text snippet to analyze

        Returns:
            List of component names that match
        """
        text_lower = text.lower()
        matched_components = []

        for component_name, config in self.components.items():
            # Check if any keywords are present
            if any(keyword.lower() in text_lower for keyword in config["keywords"]):
                matched_components.append(component_name)

        return matched_components if matched_components else ["general"]

    def categorize_texts(self, texts: List[str]) -> Dict[str, List[str]]:
        """
        Categorize a list of text segments by component

        Args:
            texts: List of text segments

        Returns:
            Dictionary mapping component names to text lists
        """
        categorized = {component: [] for component in self.components.keys()}
        categorized["general"] = []

        for text in texts:
            components = self.identify_component(text)
            for component in components:
                categorized[component].append(text)

        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}

    def get_component_weight(self, component_name: str) -> float:
        """Get the importance weight for a component"""
        return self.components.get(component_name, {}).get("weight", 1.0)

    def get_risk_keywords(self) -> List[str]:
        """Get all risk-related keywords for focused analysis"""
        return self.components["risk_factors"]["keywords"]

    def get_financial_keywords(self) -> List[str]:
        """Get all financial-related keywords"""
        return self.components["financial_performance"]["keywords"]
