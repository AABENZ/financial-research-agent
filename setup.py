"""Setup file for Financial Research Agent"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="financial-research-agent",
    version="0.1.0",
    author="Sanchit Sharma",
    author_email="sanch10.sharma@gmail.com",
    description="Multi-agent equity analysis combining SEC filings with real-time market intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SanchitSharma10/financial-research-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "lime>=0.2.0",
        "sec-edgar-downloader>=5.0.0",
        "beautifulsoup4>=4.12.0",
        "yfinance>=0.2.0",
        "newsapi-python>=0.2.7",
        "gradio>=4.0.0",
        "aiohttp>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fra-analyze=src.cli:main",
            "fra-server=src.api.server:main",
        ],
    },
)
