"""
Command-line interface for Financial Research Agent
Quick analysis without starting the web server
"""

import asyncio
import argparse
import logging
from pathlib import Path

from .core.types import AnalysisRequest
from .agents.orchestrator import EquityAnalysisOrchestrator


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def analyze_ticker(
    ticker: str,
    company_name: str = None,
    filing_type: str = "10-K",
    include_news: bool = True,
    include_technicals: bool = True,
):
    """Analyze a single ticker"""
    orchestrator = EquityAnalysisOrchestrator()

    request = AnalysisRequest(
        ticker=ticker.upper(),
        company_name=company_name,
        filing_type=filing_type,
        include_news=include_news,
        include_technicals=include_technicals,
    )

    print(f"\n{'='*60}")
    print(f"Analyzing {request.ticker}...")
    print(f"{'='*60}\n")

    result = await orchestrator.analyze(request)

    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    rec = result.recommendation

    print(f"\nTicker: {rec.ticker}")
    print(f"Sentiment: {rec.sentiment}")
    print(f"Confidence: {rec.confidence}")
    print(f"\nRecommended Action:")
    print(f"  {rec.recommended_action}")

    if rec.key_risks:
        print(f"\nKey Risks:")
        for risk in rec.key_risks:
            print(f"  [{risk.severity.upper()}] {risk.description}")

    if rec.key_opportunities:
        print(f"\nKey Opportunities:")
        for opp in rec.key_opportunities:
            print(f"  • {opp}")

    print(f"\nReasoning:")
    print("-" * 60)
    print(rec.reasoning)
    print("-" * 60)

    return result


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Financial Research Agent - Autonomous Equity Analysis"
    )

    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g., TSLA, AAPL, RBLX)",
    )

    parser.add_argument(
        "-c", "--company",
        type=str,
        help="Company name for better news search",
    )

    parser.add_argument(
        "-f", "--filing-type",
        type=str,
        default="10-K",
        choices=["10-K", "10-Q", "8-K"],
        help="SEC filing type to analyze (default: 10-K)",
    )

    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Skip news analysis",
    )

    parser.add_argument(
        "--no-technicals",
        action="store_true",
        help="Skip technical analysis",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Run analysis
    try:
        asyncio.run(
            analyze_ticker(
                ticker=args.ticker,
                company_name=args.company,
                filing_type=args.filing_type,
                include_news=not args.no_news,
                include_technicals=not args.no_technicals,
            )
        )
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
