"""
Sentiment analysis using FinBERT and SEC-BERT
Adaptive model selection based on document type
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict, Optional
import logging
from enum import Enum
import asyncio

from ..core.types import DocumentType, SentimentScore
from ..core.config import config


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available sentiment models"""
    SEC_BERT = "sec-bert"
    FINBERT = "finbert"


class SentimentAnalyzer:
    """
    Multi-model sentiment analyzer with document-type routing
    Uses SEC-BERT for SEC filings, FinBERT for news/analyst reports
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or config.model.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        self.models: Dict[str, torch.nn.Module] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.model_configs = {
            ModelType.SEC_BERT: {
                "path": config.model.sec_bert_path,
                "doc_types": [DocumentType.SEC_FILING],
            },
            ModelType.FINBERT: {
                "path": config.model.finbert_path,
                "doc_types": [DocumentType.NEWS_ARTICLE, DocumentType.ANALYST_REPORT],
            },
        }

        self._initialize_models()

    def _initialize_models(self):
        """Load all models into memory"""
        for model_type, model_config in self.model_configs.items():
            try:
                logger.info(f"Loading {model_type.value} from {model_config['path']}")
                self.tokenizers[model_type.value] = AutoTokenizer.from_pretrained(
                    model_config["path"]
                )
                self.models[model_type.value] = (
                    AutoModelForSequenceClassification.from_pretrained(
                        model_config["path"]
                    )
                    .to(self.device)
                    .eval()
                )
                logger.info(f"Successfully loaded {model_type.value}")
            except Exception as e:
                logger.error(f"Failed to load {model_type.value}: {str(e)}")
                raise

    def get_model_for_document(self, doc_type: DocumentType) -> str:
        """Select appropriate model based on document type"""
        for model_type, model_config in self.model_configs.items():
            if doc_type in model_config["doc_types"]:
                return model_type.value
        # Default to FinBERT
        return ModelType.FINBERT.value

    async def analyze_batch(
        self, texts: List[str], doc_type: DocumentType
    ) -> List[SentimentScore]:
        """
        Analyze a batch of texts asynchronously

        Args:
            texts: List of text strings to analyze
            doc_type: Type of document being analyzed

        Returns:
            List of SentimentScore objects
        """
        if not texts:
            return []

        # Flatten if nested lists
        if isinstance(texts, list) and isinstance(texts[0], list):
            texts = [item for sublist in texts for item in sublist]

        model_name = self.get_model_for_document(doc_type)
        return await self._process_batch(texts, model_name)

    async def _process_batch(
        self, texts: List[str], model_name: str
    ) -> List[SentimentScore]:
        """Process a batch with the specified model"""
        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]

            # Ensure texts are strings
            if isinstance(texts, str):
                texts = [texts]
            texts = [str(t) for t in texts]

            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.model.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Convert to SentimentScore objects
            results = []
            for probs in probabilities:
                if len(probs) == 2:  # Binary model (SEC-BERT)
                    results.append(
                        SentimentScore(
                            negative=float(probs[0]),
                            positive=float(probs[1]),
                            neutral=0.0,
                        )
                    )
                else:  # 3-class model (FinBERT)
                    results.append(
                        SentimentScore(
                            negative=float(probs[0]),
                            neutral=float(probs[1]),
                            positive=float(probs[2]),
                        )
                    )

            # Cleanup
            del inputs, outputs, probabilities
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return results

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    def predict_proba(self, texts: List[str] | str) -> np.ndarray:
        """
        Predict probabilities (for LIME compatibility)

        Args:
            texts: Text or list of texts

        Returns:
            Numpy array of shape (n_samples, n_classes)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Use SEC-BERT by default for LIME
        model_name = ModelType.SEC_BERT.value
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        # Process in batches for memory efficiency
        batch_size = 8
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())

            del inputs, outputs, probs
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Ensure 3-class output for LIME (negative, neutral, positive)
        result = np.array(all_probs)
        if result.shape[1] == 2:  # Binary model
            # Add zero neutral column
            neutral = np.zeros((result.shape[0], 1))
            result = np.hstack([result[:, 0:1], neutral, result[:, 1:2]])

        return result

    def __del__(self):
        """Cleanup on deletion"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
