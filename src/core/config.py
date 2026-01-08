"""
Configuration management for Financial Research Agent
Centralized settings for all components
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
from pathlib import Path
from dotenv import load_dotenv
import random
import numpy as np

load_dotenv()


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class ModelConfig(BaseModel):
    """Model configuration for sentiment analysis"""
    sec_bert_path: str = "nlpaueb/sec-bert-base"
    finbert_path: str = "ProsusAI/finbert"
    device: str = "cuda"  # or "cpu"
    batch_size: int = 16
    max_length: int = 512
    random_seed: int = 42  # For reproducibility
    lime_num_samples: int = 200  # Increased for stability (was 50)


class SECConfig(BaseModel):
    """SEC filing analysis configuration"""
    company_name: str = Field(default="FinancialResearchAgent")
    email: str = Field(default=os.getenv("SEC_EMAIL", "your@email.com"))
    cache_dir: str = "./sec-edgar-filings"
    filing_types: List[str] = ["10-K", "10-Q", "8-K"]
    components: List[str] = ["financial_performance", "risk_factors", "business_strategy", "operations"]


class MarketDataConfig(BaseModel):
    """Market data API configuration"""
    news_api_key: Optional[str] = Field(default=os.getenv("NEWS_API_KEY"))
    alpha_vantage_key: Optional[str] = Field(default=os.getenv("ALPHA_VANTAGE_KEY"))
    default_period: str = "1mo"
    lookback_days: int = 30


class AgentConfig(BaseModel):
    """Agent orchestration configuration"""
    llm_provider: str = "ollama"  # or "openai", "anthropic"
    model_name: str = "qwen:7b"
    base_url: Optional[str] = "http://localhost:11434"
    temperature: float = 0.7
    max_iterations: int = 3
    process_type: str = "hierarchical"  # or "sequential"


class APIConfig(BaseModel):
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True
    gradio_share: bool = True


class Config(BaseModel):
    """Main configuration object"""
    model: ModelConfig = ModelConfig()
    sec: SECConfig = SECConfig()
    market: MarketDataConfig = MarketDataConfig()
    agent: AgentConfig = AgentConfig()
    api: APIConfig = APIConfig()

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    cache_dir: Path = project_root / ".cache"
    logs_dir: Path = project_root / "logs"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


# Global config instance
config = Config()
