"""
Gradio-based web interface for equity analysis
Simple, shareable demo for Option A
"""

import gradio as gr
import asyncio
import logging
from typing import Tuple

from ..core.types import AnalysisRequest
from ..core.config import config
from ..agents.orchestrator import EquityAnalysisOrchestrator

logger = logging.getLogger(__name__)


class AnalysisServer:
    """Web interface for equity analysis"""

    def __init__(self):
        self.orchestrator = EquityAnalysisOrchestrator()

    def analyze(
        self,
        ticker: str,
        company_name: str = "",
        filing_type: str = "10-K",
        include_news: bool = True,
        include_technicals: bool = True,
    ) -> Tuple[str, str, str, str, str]:
        """
        Main analysis function for Gradio interface

        Returns:
            Tuple of (sentiment, confidence, risks, opportunities, recommendation, reasoning)
        """
        try:
            # Validate input
            if not ticker:
                return ("Error", "N/A", "Ticker symbol required", "N/A", "N/A", "N/A")

            ticker = ticker.strip().upper()

            # Create request
            request = AnalysisRequest(
                ticker=ticker,
                company_name=company_name or None,
                filing_type=filing_type,
                include_news=include_news,
                include_technicals=include_technicals,
            )

            # Run analysis
            result = asyncio.run(self.orchestrator.analyze(request))

            # Extract recommendation
            rec = result.recommendation

            # Format risks
            risks_text = "\n".join(
                [
                    f"• [{r.severity.upper()}] {r.description}"
                    for r in rec.key_risks
                ]
            )

            # Format opportunities
            opps_text = "\n".join(
                [f"• {opp}" for opp in rec.key_opportunities]
            )

            return (
                rec.sentiment,
                rec.confidence,
                risks_text if risks_text else "No significant risks identified",
                opps_text if opps_text else "No significant opportunities identified",
                rec.recommended_action,
                rec.reasoning,
            )

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return ("Error", "N/A", f"Analysis failed: {str(e)}", "N/A", "N/A", "N/A")

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""

        with gr.Blocks(title="Financial Research Agent") as interface:
            gr.Markdown("# 📊 Financial Research Agent")
            gr.Markdown(
                "Multi-agent equity analysis combining SEC filings with real-time market intelligence"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    ticker_input = gr.Textbox(
                        label="Ticker Symbol",
                        placeholder="e.g., TSLA, AAPL, RBLX",
                        value="RBLX",
                    )
                    company_input = gr.Textbox(
                        label="Company Name (Optional)",
                        placeholder="e.g., Roblox Corporation",
                    )
                    filing_type = gr.Dropdown(
                        choices=["10-K", "10-Q", "8-K"],
                        value="10-K",
                        label="SEC Filing Type",
                    )

                    with gr.Row():
                        include_news = gr.Checkbox(
                            label="Include News Analysis", value=True
                        )
                        include_technicals = gr.Checkbox(
                            label="Include Technical Analysis", value=True
                        )

                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Results")

                    sentiment = gr.Textbox(label="Market Sentiment", interactive=False)
                    confidence = gr.Textbox(label="Confidence Level", interactive=False)
                    risks = gr.Textbox(
                        label="Key Risks",
                        lines=5,
                        interactive=False,
                    )
                    opportunities = gr.Textbox(
                        label="Key Opportunities",
                        lines=3,
                        interactive=False,
                    )
                    action = gr.Textbox(
                        label="Recommended Action",
                        lines=2,
                        interactive=False,
                    )
                    reasoning = gr.Textbox(
                        label="Detailed Analysis",
                        lines=8,
                        interactive=False,
                    )

            # Examples
            gr.Examples(
                examples=[
                    ["TSLA", "Tesla Inc", "10-K", True, True],
                    ["AAPL", "Apple Inc", "10-Q", True, True],
                    ["RBLX", "Roblox Corporation", "10-K", True, True],
                ],
                inputs=[
                    ticker_input,
                    company_input,
                    filing_type,
                    include_news,
                    include_technicals,
                ],
            )

            # Connect analysis function
            analyze_btn.click(
                fn=self.analyze,
                inputs=[
                    ticker_input,
                    company_input,
                    filing_type,
                    include_news,
                    include_technicals,
                ],
                outputs=[sentiment, confidence, risks, opportunities, action, reasoning],
            )

            gr.Markdown("---")
            gr.Markdown(
                """
                ### How it works:
                1. **SEC Filing Agent** - Analyzes SEC filings with FinBERT/SEC-BERT + LIME explainability
                2. **Market Intelligence Agent** - Gathers real-time price data, technicals, and news (Gemini-powered sentiment)
                3. **Synthesis Agent** - Gemini 2.0 Flash synthesizes fundamental vs. market data for final recommendation

                Built with multi-agent architecture for deep, explainable equity analysis. Combines domain-specific ML models with LLM reasoning.
                """
            )

        return interface

    def launch(self, share: bool = True, debug: bool = False):
        """Launch the Gradio server"""
        interface = self.create_interface()
        interface.launch(
            server_name=config.api.host,
            server_port=config.api.port,
            share=share,
            debug=debug,
        )


def main():
    """Entry point for running the server"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = AnalysisServer()
    logger.info("Starting Financial Research Agent server...")
    server.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
