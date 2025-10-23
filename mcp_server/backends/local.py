"""
Local backend - runs TTS model in-process.

Supports two modes:
1. Lazy loading: Load model on first request, unload after timeout
2. Always loaded: Keep model in memory

Memory usage: ~8GB when loaded, ~100MB when unloaded
"""

import asyncio
import logging
import time
import io
from typing import Dict, Any, Optional, Literal
from pathlib import Path
import torch
import torchaudio as ta

from .base import TTSBackend, TTSRequest, TTSResponse, BackendError

logger = logging.getLogger(__name__)


class LocalBackend(TTSBackend):
    """
    Runs Chatterbox TTS model locally.

    Features:
    - Lazy loading: Load on first use, unload after idle timeout
    - Always loaded: Keep in memory
    - Device selection: CPU, CUDA, MPS
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        model_type: Literal["english", "multilingual"] = "english",
        lazy_mode: bool = True,
        unload_timeout: int = 600,  # 10 minutes
        max_concurrent: int = 1,
    ):
        """
        Args:
            device: Compute device (cpu/cuda/mps)
            model_type: Model variant (english/multilingual)
            lazy_mode: Enable lazy loading (unload after timeout)
            unload_timeout: Seconds before unloading idle model
            max_concurrent: Maximum concurrent generations
        """
        self.device = device
        self.model_type = model_type
        self.lazy_mode = lazy_mode
        self.unload_timeout = unload_timeout
        self.max_concurrent = max_concurrent

        # Model instance (loaded on demand)
        self._model = None
        self._last_used = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Background task for auto-unload
        self._unload_task = None
        if lazy_mode:
            self._start_unload_watcher()

        logger.info(
            f"LocalBackend initialized: device={device}, "
            f"model={model_type}, lazy={lazy_mode}"
        )

    def _start_unload_watcher(self):
        """Start background task to auto-unload idle model."""
        async def watch():
            while True:
                await asyncio.sleep(60)  # Check every minute
                if self._model and self._last_used:
                    idle_time = time.time() - self._last_used
                    if idle_time > self.unload_timeout:
                        logger.info(
                            f"Model idle for {idle_time:.0f}s, unloading..."
                        )
                        await self._unload_model()

        self._unload_task = asyncio.create_task(watch())

    async def _load_model(self):
        """Load TTS model into memory."""
        async with self._lock:
            if self._model is not None:
                return  # Already loaded

            logger.info(f"Loading {self.model_type} model on {self.device}...")
            start = time.time()

            # Import here to avoid loading heavy dependencies on startup
            if self.model_type == "english":
                from chatterbox.tts import ChatterboxTTS
                self._model = await asyncio.to_thread(
                    ChatterboxTTS.from_pretrained,
                    device=self.device
                )
            else:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self._model = await asyncio.to_thread(
                    ChatterboxMultilingualTTS.from_pretrained,
                    device=self.device
                )

            elapsed = time.time() - start
            logger.info(f"Model loaded in {elapsed:.1f}s")

    async def _unload_model(self):
        """Unload model from memory to free resources."""
        async with self._lock:
            if self._model is None:
                return

            logger.info("Unloading model...")
            self._model = None

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded, memory freed")

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate audio using local model."""
        # Limit concurrent generations
        async with self._semaphore:
            try:
                # Ensure model is loaded
                if self._model is None:
                    await self._load_model()

                # Update last used time
                self._last_used = time.time()

                # Prepare voice conditionals if needed
                if request.voice_path:
                    await asyncio.to_thread(
                        self._model.prepare_conditionals,
                        request.voice_path,
                        exaggeration=request.exaggeration
                    )

                # Build generation kwargs
                gen_kwargs = {
                    "text": request.text,
                    "exaggeration": request.exaggeration,
                    "cfg_weight": request.cfg_weight,
                    "temperature": request.temperature,
                    "repetition_penalty": request.repetition_penalty,
                    "min_p": request.min_p,
                    "top_p": request.top_p,
                }

                # Add language_id for multilingual
                if request.language_id and self.model_type == "multilingual":
                    gen_kwargs["language_id"] = request.language_id

                # Generate audio
                logger.info(f"Generating audio for {len(request.text)} chars...")
                start = time.time()

                wav = await asyncio.to_thread(
                    self._model.generate,
                    **gen_kwargs
                )

                elapsed = time.time() - start
                duration = wav.shape[-1] / self._model.sr

                logger.info(
                    f"Generated {duration:.1f}s audio in {elapsed:.1f}s "
                    f"(RTF: {elapsed/duration:.2f}x)"
                )

                # Convert to WAV bytes
                buffer = io.BytesIO()
                await asyncio.to_thread(
                    ta.save,
                    buffer,
                    wav,
                    self._model.sr,
                    format="wav"
                )
                audio_data = buffer.getvalue()

                return TTSResponse(
                    audio_data=audio_data,
                    sample_rate=self._model.sr,
                    duration_seconds=duration,
                    metadata={
                        "backend": "local",
                        "device": self.device,
                        "model_type": self.model_type,
                        "generation_time": elapsed,
                        "rtf": elapsed / duration,
                    }
                )

            except Exception as e:
                logger.exception("Generation failed")
                raise BackendError(f"Local generation failed: {e}")

    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            # Check if device is available
            if self.device == "cuda" and not torch.cuda.is_available():
                return False
            if self.device == "mps" and not torch.backends.mps.is_available():
                return False
            return True
        except:
            return False

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        is_loaded = self._model is not None
        is_healthy = await self.health_check()

        info = {
            "backend_type": "local",
            "device": self.device,
            "model_type": self.model_type,
            "lazy_mode": self.lazy_mode,
            "model_loaded": is_loaded,
            "healthy": is_healthy,
            "max_concurrent": self.max_concurrent,
        }

        if is_loaded and self._last_used:
            info["idle_time"] = time.time() - self._last_used

        return info

    async def cleanup(self):
        """Cleanup resources."""
        # Cancel unload watcher
        if self._unload_task:
            self._unload_task.cancel()

        # Unload model
        await self._unload_model()

        logger.info("LocalBackend cleaned up")
