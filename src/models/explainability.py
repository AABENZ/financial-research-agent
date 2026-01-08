"""
LIME-based explainability for sentiment predictions
Shows which words/phrases drive model decisions
"""

from lime.lime_text import LimeTextExplainer
from typing import List, Dict, Callable, Optional
import numpy as np
import logging
import gc

from ..core.types import WordImportance
from ..core.config import config

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Provides interpretable explanations for sentiment predictions
    Uses LIME to identify influential words and phrases
    """

    def __init__(self, predict_fn: Callable):
        """
        Initialize explainability engine

        Args:
            predict_fn: Function that takes texts and returns probability arrays
                       Should have signature: fn(List[str]) -> np.ndarray
        """
        self.predict_fn = predict_fn
        self.explainer = LimeTextExplainer(
            class_names=["negative", "neutral", "positive"],
            random_state=config.model.random_seed  # For reproducibility
        )

    def explain(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 500,
    ) -> List[WordImportance]:
        """
        Generate explanation for a text prediction

        Args:
            text: Text to explain
            num_features: Number of important features to return
            num_samples: Number of samples for LIME (higher = more accurate but slower)

        Returns:
            List of WordImportance objects showing influential words
        """
        try:
            # Generate explanation
            exp = self.explainer.explain_instance(
                text, self.predict_fn, num_features=num_features, num_samples=num_samples
            )

            # Extract word importance
            word_scores = exp.as_list()

            # Convert to WordImportance objects
            explanations = [
                WordImportance(
                    word=word,
                    importance=score,
                    sentiment="positive" if score > 0 else "negative",
                    magnitude=abs(score),
                )
                for word, score in word_scores
            ]

            # Cleanup
            del exp
            gc.collect()

            return explanations

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return []

    def explain_batch(
        self,
        texts: List[str],
        sentiments: List[np.ndarray],
        threshold: float = 0.6,
        max_explanations: int = 10,
        num_features: int = 8,
    ) -> List[Dict]:
        """
        Generate explanations for significant predictions in a batch

        Args:
            texts: List of texts
            sentiments: List of sentiment probability arrays
            threshold: Only explain predictions with confidence above this
            max_explanations: Maximum number of explanations to generate
            num_features: Number of features per explanation

        Returns:
            List of explanation dictionaries
        """
        explanations = []
        count = 0

        for text, sentiment in zip(texts, sentiments):
            # Only explain high-confidence predictions
            if max(sentiment) > threshold and count < max_explanations:
                try:
                    word_importance = self.explain(text, num_features=num_features, num_samples=50)
                    explanations.append(
                        {
                            "text": text,
                            "sentiment": sentiment.tolist(),
                            "important_words": [wi.dict() for wi in word_importance],
                        }
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to explain text: {str(e)}")
                    continue

        return explanations

    def get_top_phrases(
        self, explanations: List[Dict], sentiment_filter: Optional[str] = None, top_k: int = 5
    ) -> List[str]:
        """
        Extract top influential phrases across all explanations

        Args:
            explanations: List of explanation dictionaries
            sentiment_filter: Only include "positive" or "negative" words (None = both)
            top_k: Number of top phrases to return

        Returns:
            List of most influential phrases
        """
        word_scores = {}

        for exp in explanations:
            for word_info in exp.get("important_words", []):
                word = word_info["word"]
                importance = word_info["importance"]
                sentiment = word_info["sentiment"]

                # Apply filter if specified
                if sentiment_filter and sentiment != sentiment_filter:
                    continue

                # Accumulate scores
                if word not in word_scores:
                    word_scores[word] = 0
                word_scores[word] += abs(importance)

        # Sort and return top k
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words[:top_k]]
