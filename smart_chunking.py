"""
Smart text chunking for long-form TTS generation.

Intelligently splits text at natural boundaries (sentences, clauses, paragraphs)
to maintain prosody and avoid disjointed speech.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text with metadata for TTS processing."""
    text: str
    chunk_type: str  # 'sentence', 'clause', 'paragraph'
    is_final_in_paragraph: bool = False
    is_final_in_document: bool = False


class SmartChunker:
    """
    Intelligently chunk text for TTS while maintaining natural speech flow.

    Strategy:
    1. Split on paragraph boundaries (double newlines)
    2. Within paragraphs, split on sentence boundaries (. ! ?)
    3. If sentences are too long, split on clause boundaries (, ; :)
    4. Preserve context for proper intonation
    """

    def __init__(self, max_chars: int = 250, min_chars: int = 50):
        """
        Args:
            max_chars: Target maximum characters per chunk (soft limit)
            min_chars: Minimum characters to avoid too-short chunks
        """
        self.max_chars = max_chars
        self.min_chars = min_chars

        # Sentence boundaries
        self.sentence_pattern = re.compile(r'([.!?]+(?:\s|$))')

        # Clause boundaries (secondary split points)
        self.clause_pattern = re.compile(r'([,;:]\s)')

        # Paragraph boundaries
        self.paragraph_pattern = re.compile(r'\n\s*\n')

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks suitable for TTS generation.

        Returns:
            List of TextChunk objects with proper boundary markers
        """
        if not text or len(text.strip()) == 0:
            return []

        # Split into paragraphs first
        paragraphs = self.paragraph_pattern.split(text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []

        for para_idx, paragraph in enumerate(paragraphs):
            is_final_paragraph = (para_idx == len(paragraphs) - 1)
            para_chunks = self._chunk_paragraph(paragraph, is_final_paragraph)
            chunks.extend(para_chunks)

        # Mark the final chunk in the document
        if chunks:
            chunks[-1].is_final_in_document = True

        return chunks

    def _chunk_paragraph(self, paragraph: str, is_final_paragraph: bool) -> List[TextChunk]:
        """Chunk a single paragraph into TTS-friendly pieces."""

        # If paragraph is short enough, return as-is
        if len(paragraph) <= self.max_chars:
            return [TextChunk(
                text=paragraph,
                chunk_type='paragraph',
                is_final_in_paragraph=True,
                is_final_in_document=is_final_paragraph
            )]

        # Split into sentences
        sentences = self._split_sentences(paragraph)

        chunks = []
        current_chunk = []
        current_length = 0

        for sent_idx, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            # If single sentence is too long, split on clauses
            if sentence_len > self.max_chars:
                # Flush current chunk if exists
                if current_chunk:
                    chunks.append(TextChunk(
                        text=' '.join(current_chunk),
                        chunk_type='sentence',
                        is_final_in_paragraph=False
                    ))
                    current_chunk = []
                    current_length = 0

                # Split long sentence into clauses
                clause_chunks = self._split_long_sentence(sentence)
                chunks.extend(clause_chunks)
                continue

            # Check if adding this sentence would exceed max_chars
            if current_length + sentence_len > self.max_chars and current_chunk:
                # Flush current chunk
                chunks.append(TextChunk(
                    text=' '.join(current_chunk),
                    chunk_type='sentence',
                    is_final_in_paragraph=False
                ))
                current_chunk = []
                current_length = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space

        # Flush remaining chunk
        if current_chunk:
            chunks.append(TextChunk(
                text=' '.join(current_chunk),
                chunk_type='sentence',
                is_final_in_paragraph=True,
                is_final_in_document=is_final_paragraph
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving sentence-ending punctuation."""
        parts = self.sentence_pattern.split(text)

        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentence = (parts[i] + parts[i + 1]).strip()
                if sentence:
                    sentences.append(sentence)

        # Handle any remaining text without sentence-ending punctuation
        if parts and parts[-1].strip():
            sentences.append(parts[-1].strip())

        return sentences

    def _split_long_sentence(self, sentence: str) -> List[TextChunk]:
        """Split a long sentence on clause boundaries."""
        # Split on commas, semicolons, colons
        parts = self.clause_pattern.split(sentence)

        chunks = []
        current_chunk = []
        current_length = 0

        for i in range(0, len(parts), 2):
            clause = parts[i]
            separator = parts[i + 1] if i + 1 < len(parts) else ''

            clause_text = clause + separator
            clause_len = len(clause_text)

            # If even a single clause is too long, just include it
            # (better than cutting mid-word)
            if current_length + clause_len > self.max_chars and current_chunk:
                chunks.append(TextChunk(
                    text=''.join(current_chunk).strip(),
                    chunk_type='clause',
                    is_final_in_paragraph=False
                ))
                current_chunk = []
                current_length = 0

            current_chunk.append(clause_text)
            current_length += clause_len

        # Flush remaining
        if current_chunk:
            chunks.append(TextChunk(
                text=''.join(current_chunk).strip(),
                chunk_type='clause',
                is_final_in_paragraph=False
            ))

        return chunks

    def estimate_duration(self, text: str, words_per_minute: float = 150) -> float:
        """
        Estimate audio duration in seconds.

        Args:
            text: Input text
            words_per_minute: Average speaking rate (default 150 WPM)

        Returns:
            Estimated duration in seconds
        """
        word_count = len(text.split())
        duration = (word_count / words_per_minute) * 60
        return duration


def demonstrate_chunking():
    """Example usage of smart chunking."""

    sample_text = """
    Chatterbox is a production-grade, multilingual TTS system that supports 23 languages.
    It uses a two-stage pipeline: first, the T3 model converts text into discrete speech tokens
    using a Llama 3.1 backbone with 520 million parameters. Then, the S3Gen model uses
    flow-matching diffusion to convert those tokens into high-quality 24kHz audio.

    The system is unique because it offers emotion exaggeration control, allowing you to
    adjust the intensity of speech from neutral (0.5) to highly expressive (2.0). It also
    supports zero-shot voice cloning, meaning you can clone any voice from just a short
    reference sample, without retraining the model.

    For deployment, you'll want to implement smart chunking to handle long documents.
    This ensures natural speech flow by splitting text at sentence and clause boundaries,
    rather than arbitrary character limits. The result is much more natural-sounding audio
    for long-form content like essays, articles, or audiobooks.
    """

    chunker = SmartChunker(max_chars=200, min_chars=50)
    chunks = chunker.chunk_text(sample_text)

    print("=" * 80)
    print("SMART CHUNKING DEMONSTRATION")
    print("=" * 80)

    for i, chunk in enumerate(chunks, 1):
        duration = chunker.estimate_duration(chunk.text)
        print(f"\n--- Chunk {i} ({chunk.chunk_type}) ---")
        print(f"Length: {len(chunk.text)} chars")
        print(f"Est. Duration: {duration:.1f}s")
        print(f"Final in paragraph: {chunk.is_final_in_paragraph}")
        print(f"Final in document: {chunk.is_final_in_document}")
        print(f"Text: {chunk.text}")

    print(f"\n{'=' * 80}")
    print(f"Total chunks: {len(chunks)}")
    total_duration = sum(chunker.estimate_duration(c.text) for c in chunks)
    print(f"Estimated total audio duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")


if __name__ == "__main__":
    demonstrate_chunking()
