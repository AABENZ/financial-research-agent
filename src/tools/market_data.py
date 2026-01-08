"""
Market data tool using yfinance
Provides real-time price, volume, and technical indicators
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict
from datetime import datetime
import logging

from ..core.types import MarketData
from ..core.config import config

logger = logging.getLogger(__name__)


class MarketDataTool:
    """
    Fetches and processes market data for equities
    - Current price and changes
    - Volume analysis
    - Technical indicators (RSI, MACD, Moving Averages)
    """

    def __init__(self):
        self.default_period = config.market.default_period

    def get_market_data(
        self, ticker: str, period: str = "1mo"
    ) -> Optional[MarketData]:
        """
        Get comprehensive market data for a ticker

        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, etc.)

        Returns:
            MarketData object or None if failed
        """
        try:
            logger.info(f"Fetching market data for {ticker}, period: {period}")

            # Download data with retry and proper headers
            stock = yf.Ticker(ticker)

            # Try with different methods to avoid rate limiting
            try:
                # Method 1: Standard history
                df = stock.history(period=period)
            except:
                # Method 2: Use download with proper headers
                import time
                time.sleep(1)  # Brief delay to avoid rate limiting
                df = yf.download(ticker, period=period, progress=False, show_errors=False)

            if df.empty:
                logger.warning(f"No data available for {ticker}")
                return None

            # Current price metrics
            current_price = float(df["Close"].iloc[-1])
            start_price = float(df["Close"].iloc[0])
            price_change = current_price - start_price
            price_change_pct = (price_change / start_price) * 100

            # Calculate technical indicators
            rsi = self._calculate_rsi(df["Close"])
            macd_data = self._calculate_macd(df["Close"])
            moving_avgs = self._calculate_moving_averages(df["Close"])

            # Volume
            avg_volume = int(df["Volume"].mean())

            return MarketData(
                ticker=ticker,
                current_price=current_price,
                price_change=price_change,
                price_change_pct=price_change_pct,
                volume=avg_volume,
                rsi=rsi,
                macd=macd_data,
                moving_averages=moving_avgs,
            )

        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {str(e)}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        except Exception as e:
            logger.warning(f"Error calculating RSI: {str(e)}")
            return None

    def _calculate_macd(
        self, prices: pd.Series
    ) -> Optional[Dict[str, float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal

            return {
                "macd": float(macd.iloc[-1]),
                "signal": float(signal.iloc[-1]),
                "histogram": float(histogram.iloc[-1]),
            }
        except Exception as e:
            logger.warning(f"Error calculating MACD: {str(e)}")
            return None

    def _calculate_moving_averages(
        self, prices: pd.Series
    ) -> Optional[Dict[str, float]]:
        """Calculate moving averages"""
        try:
            ma_periods = [20, 50, 200]
            mas = {}

            for period in ma_periods:
                if len(prices) >= period:
                    ma = prices.rolling(window=period).mean()
                    mas[f"ma_{period}"] = float(ma.iloc[-1])

            return mas if mas else None
        except Exception as e:
            logger.warning(f"Error calculating moving averages: {str(e)}")
            return None

    def get_price_trend(self, ticker: str, period: str = "1mo") -> Optional[str]:
        """
        Determine price trend (BULLISH/BEARISH/NEUTRAL)

        Args:
            ticker: Stock ticker
            period: Time period

        Returns:
            Trend string or None
        """
        market_data = self.get_market_data(ticker, period)
        if not market_data:
            return None

        # Simple trend based on price change and moving averages
        if market_data.price_change_pct > 5:
            return "BULLISH"
        elif market_data.price_change_pct < -5:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def get_technical_signals(self, ticker: str) -> Dict[str, str]:
        """
        Get technical indicator signals

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary of indicator signals
        """
        market_data = self.get_market_data(ticker)
        if not market_data:
            return {}

        signals = {}

        # RSI signal
        if market_data.rsi:
            if market_data.rsi > 70:
                signals["rsi"] = "OVERBOUGHT"
            elif market_data.rsi < 30:
                signals["oversold"] = "OVERSOLD"
            else:
                signals["rsi"] = "NEUTRAL"

        # MACD signal
        if market_data.macd:
            if market_data.macd["histogram"] > 0:
                signals["macd"] = "BULLISH"
            else:
                signals["macd"] = "BEARISH"

        # Moving average signal
        if market_data.moving_averages:
            if "ma_20" in market_data.moving_averages and "ma_50" in market_data.moving_averages:
                if market_data.moving_averages["ma_20"] > market_data.moving_averages["ma_50"]:
                    signals["ma_trend"] = "BULLISH"
                else:
                    signals["ma_trend"] = "BEARISH"

        return signals
