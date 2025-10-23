"""
Long-form TTS generation with smart chunking.

Handles arbitrarily large text by:
1. Intelligently chunking at sentence/paragraph boundaries
2. Maintaining voice consistency across chunks
3. Smooth audio concatenation with crossfading
"""

import torch
import torchaudio as ta
from typing import Optional, List
from pathlib import Path

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from smart_chunking import SmartChunker, TextChunk


class LongFormTTS:
    """
    TTS system optimized for long-form content (essays, articles, audiobooks).

    Features:
    - Smart text chunking at natural boundaries
    - Voice consistency across chunks
    - Smooth concatenation with crossfading to avoid clicks
    - Progress tracking for long documents
    """

    def __init__(
        self,
        model: ChatterboxTTS,
        max_chunk_chars: int = 250,
        crossfade_samples: int = 480  # 20ms at 24kHz
    ):
        """
        Args:
            model: Initialized ChatterboxTTS or ChatterboxMultilingualTTS
            max_chunk_chars: Maximum characters per chunk
            crossfade_samples: Number of samples for crossfading between chunks
        """
        self.model = model
        self.chunker = SmartChunker(max_chars=max_chunk_chars)
        self.crossfade_samples = crossfade_samples

        # Pre-compute crossfade envelope (cosine curve)
        fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
        fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
        self.fade_out = 0.5 * (1 + torch.cos(torch.pi * (1 - fade_out)))
        self.fade_in = 0.5 * (1 + torch.cos(torch.pi * (1 - fade_in)))

    def generate_long_form(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        language_id: Optional[str] = None,  # For multilingual model
        show_progress: bool = True,
        save_chunks: bool = False,  # For debugging
        output_dir: Optional[Path] = None,
    ) -> torch.Tensor:
        """
        Generate audio for long-form text with smart chunking.

        Args:
            text: Input text (can be arbitrarily long)
            audio_prompt_path: Reference voice sample path
            exaggeration: Emotional intensity (0.25-2.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling temperature
            language_id: Language code for multilingual model (e.g., 'fr', 'es')
            show_progress: Print progress during generation
            save_chunks: Save individual chunks for debugging
            output_dir: Directory to save chunks (if save_chunks=True)

        Returns:
            Concatenated audio tensor (1, samples)
        """

        # Prepare voice conditionals once (reused across all chunks)
        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

        # Chunk the text
        chunks = self.chunker.chunk_text(text)

        if show_progress:
            total_duration = sum(self.chunker.estimate_duration(c.text) for c in chunks)
            print(f"Processing {len(chunks)} chunks (~{total_duration:.0f}s of audio)")

        # Generate audio for each chunk
        audio_chunks = []

        for i, chunk in enumerate(chunks, 1):
            if show_progress:
                duration = self.chunker.estimate_duration(chunk.text)
                print(f"[{i}/{len(chunks)}] Generating chunk ({len(chunk.text)} chars, ~{duration:.1f}s)...")

            # Generate audio for this chunk
            generation_kwargs = {
                'text': chunk.text,
                'exaggeration': exaggeration,
                'cfg_weight': cfg_weight,
                'temperature': temperature,
            }

            # Add language_id if using multilingual model
            if language_id and hasattr(self.model, 'generate'):
                if 'language_id' in self.model.generate.__code__.co_varnames:
                    generation_kwargs['language_id'] = language_id

            wav = self.model.generate(**generation_kwargs)

            # Optionally save individual chunks
            if save_chunks and output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                chunk_path = output_dir / f"chunk_{i:03d}.wav"
                ta.save(str(chunk_path), wav, self.model.sr)
                if show_progress:
                    print(f"  Saved: {chunk_path}")

            audio_chunks.append(wav)

        if show_progress:
            print("Concatenating chunks with crossfading...")

        # Concatenate with crossfading
        final_audio = self._concatenate_with_crossfade(audio_chunks)

        if show_progress:
            duration = final_audio.shape[-1] / self.model.sr
            print(f"Done! Total audio duration: {duration:.1f}s")

        return final_audio

    def _concatenate_with_crossfade(self, audio_chunks: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate audio chunks with smooth crossfading to avoid clicks.

        Strategy:
        - Fade out end of chunk N
        - Fade in start of chunk N+1
        - Overlap the faded regions
        """
        if not audio_chunks:
            return torch.zeros(1, 0)

        if len(audio_chunks) == 1:
            return audio_chunks[0]

        result = audio_chunks[0]

        for next_chunk in audio_chunks[1:]:
            # Ensure chunks are at least as long as crossfade
            if result.shape[-1] < self.crossfade_samples:
                # Too short to crossfade, just concatenate
                result = torch.cat([result, next_chunk], dim=-1)
                continue

            if next_chunk.shape[-1] < self.crossfade_samples:
                # Next chunk too short, just append
                result = torch.cat([result, next_chunk], dim=-1)
                continue

            # Extract crossfade regions
            fade_out_region = result[:, -self.crossfade_samples:].clone()
            fade_in_region = next_chunk[:, :self.crossfade_samples].clone()

            # Apply crossfade envelopes
            fade_out_region *= self.fade_out
            fade_in_region *= self.fade_in

            # Mix the overlapping regions
            mixed_region = fade_out_region + fade_in_region

            # Concatenate: result[:-fade] + mixed + next[fade:]
            result = torch.cat([
                result[:, :-self.crossfade_samples],
                mixed_region,
                next_chunk[:, self.crossfade_samples:]
            ], dim=-1)

        return result


def example_long_form_generation():
    """Example: Generate audio for a long article."""

    # Automatically detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}\n")

    # Load model
    print("Loading Chatterbox TTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)

    # Initialize long-form TTS system
    long_form = LongFormTTS(model, max_chunk_chars=200)

    # Example long text (article about TTS)
    article = """
    Text-to-speech technology has revolutionized how we interact with computers and consume content.
    Modern neural TTS systems can generate remarkably natural-sounding speech that's nearly
    indistinguishable from human recordings.

    The latest generation of TTS models uses transformer architectures and diffusion techniques
    to produce high-quality audio. These systems can clone voices from just a few seconds of
    reference audio, making them incredibly versatile for different applications.

    One of the most exciting developments is emotion control. Users can now adjust the
    expressiveness of synthesized speech, from neutral delivery to highly dramatic performances.
    This opens up new possibilities for creative content, audiobooks, and interactive experiences.

    For developers building TTS applications, it's important to consider how to handle long-form
    content efficiently. Smart chunking strategies that split text at natural boundaries like
    sentences and paragraphs ensure that the synthesized speech maintains proper prosody and
    intonation throughout.

    As these technologies continue to improve, we're moving closer to a future where synthetic
    voices are indistinguishable from real ones, while remaining accessible and customizable
    for everyone.
    """

    # Generate audio
    output_path = Path("long_form_example.wav")

    print("\nGenerating long-form audio...\n")
    audio = long_form.generate_long_form(
        text=article,
        exaggeration=0.6,  # Slightly expressive
        cfg_weight=0.5,
        temperature=0.8,
        show_progress=True,
        save_chunks=False,  # Set to True to save individual chunks
        output_dir=Path("./chunks")
    )

    # Save final audio
    ta.save(str(output_path), audio, model.sr)
    print(f"\nSaved audio to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    example_long_form_generation()
