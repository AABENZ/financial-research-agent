"""
Text extraction from SEC filings
Focuses on narrative sections while filtering boilerplate
"""

import bs4
from typing import Dict, List, Optional
import logging

from .component_analyzer import ComponentAnalyzer

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Extracts meaningful text from SEC filings
    Filters out boilerplate and focuses on key narrative sections
    """

    def __init__(self):
        self.component_analyzer = ComponentAnalyzer()

        # Sections to exclude (boilerplate)
        self.exclude_sections = [
            "forward-looking statements",
            "safe harbor",
            "market value",
            "common stock",
            "pursuant to",
            "form 10-k",
            "table of contents",
            "exhibits",
            "signatures",
            "index",
        ]

        # Minimum text length to consider
        self.min_text_length = 200

    def extract_from_file(self, file_path: str) -> Dict[str, List[str]]:
        """
        Extract text from SEC filing HTML/TXT file

        Args:
            file_path: Path to the filing file

        Returns:
            Dictionary mapping component names to lists of text segments
        """
        try:
            logger.info(f"Extracting text from: {file_path}")

            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()

            # Parse HTML
            soup = bs4.BeautifulSoup(content, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "table"]):
                element.decompose()

            # Extract text by component
            text_by_component = {}

            # Process paragraphs and divs
            for section in soup.find_all(["div", "p"]):
                text = section.get_text().strip()

                # Skip if too short or contains excluded content
                if len(text) < self.min_text_length:
                    continue

                if any(exclude in text.lower() for exclude in self.exclude_sections):
                    continue

                # Skip if mostly numbers (financial tables)
                if sum(c.isdigit() for c in text) / len(text) > 0.3:
                    continue

                # Identify component(s) for this text
                components = self.component_analyzer.identify_component(text)

                for component_name in components:
                    if component_name not in text_by_component:
                        text_by_component[component_name] = []
                    text_by_component[component_name].append(text)

            logger.info(f"Extracted text for {len(text_by_component)} components")
            for comp, texts in text_by_component.items():
                logger.info(f"  {comp}: {len(texts)} segments")

            return text_by_component

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {}

    def chunk_text(self, text: str, chunk_size: int = 2048) -> List[str]:
        """
        Split text into chunks while preserving sentence boundaries

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 for the period

            if current_length + sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks

    def chunk_texts(self, texts: List[str], chunk_size: int = 2048) -> List[str]:
        """
        Chunk multiple texts into uniform segments

        Args:
            texts: List of text strings
            chunk_size: Maximum chunk size

        Returns:
            Flattened list of text chunks
        """
        all_chunks = []
        for text in texts:
            chunks = self.chunk_text(text, chunk_size)
            all_chunks.extend(chunks)
        return all_chunks
